import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from operator import mul
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import feeders
import random
import utility
from functools import reduce

class psvrt(feeders.Feeder):

    def initialize_vars(self,
                        problem_type, item_size, box_extent,
                        num_items=2, num_item_pixel_values=1, SD_portion=0, SR_portion=1, SR_type = 'average_orientation',
                        perforate_type=None, perforate_mask=None, perforate=False,
                        position_sampler=None, item_sampler=None, mix_rules=False,
                        display=False,
                        easy=False):
        self.problem_type = problem_type
        self.item_size = item_size
        if len(item_size) == 2:
            self.item_size += [self.raw_input_size[2]]

        self.box_extent = box_extent
        self.num_items = num_items
        self.num_item_pixel_values = num_item_pixel_values
        self.SD_portion = SD_portion
        self.SR_portion = SR_portion
        self.SR_type = SR_type
        self.perforate_type = perforate_type
        self.perforate = perforate
        self.position_sampler = position_sampler
        self.item_sampler = item_sampler
        self.mix_rules = mix_rules
        self.display = display
        self.easy = easy
        if self.perforate_type is not None:
            _, clean_tracker_mask, clean_perforate_mask = self.perforate_type(None, None, self.box_extent, self.item_size,
                                                                              self.num_item_pixel_values,
                                                                              None, None)
            self.tracker_mask = clean_tracker_mask
            if perforate_mask is not None:
                self.perforate_mask = perforate_mask
            else:
                self.perforate_mask = clean_perforate_mask

        else:
            self.tracker_mask = None
            self.perforate_mask = None

    def single_batch(self, position_sampler_args=None, item_sampler_args=None, label_batch=None):
        # 1.
        #   if sd_portion = 0.5, patch = 2x2, # different pixels = 2,
        #   it's categorized as 'SAME', POSITIVE
        # 2.
        #   if sp_portion = 1 and the two patches lie along a diagonal,
        #   it's categorized as 'UD', POSITIVE

        input_data = np.zeros(dtype=np.float32, shape=(self.batch_size,) + tuple(self.raw_input_size))
        target_output = np.zeros(dtype=np.float32, shape=(self.batch_size, 1, 1, 2))
        if label_batch is None:
            label_batch = np.random.randint(low=0, high=2, size=(self.batch_size))
            # print(label_batch)
        elif label_batch.shape[0] != (self.batch_size):
            raise ValueError('label_batch is not correctly batch sized.')

        positions_list_batch = []
        items_list_batch = []
#        if self.num_items != 2:
#            raise ValueError('Num items other than 2 not implemented yet.')

        iimage = 0
        while iimage < self.batch_size:
            if self.position_sampler is None:
                # sample positions
                positions_list, SR_label = self.sample_positions(SR_label=label_batch[iimage] if (self.problem_type == 'SR') else None,
                                                       SR_portion=self.SR_portion, SR_type = self.SR_type)

                # print(positions_list)
                # print(SR_label)
                # exit(0)
            else:
                positions_list, SR_label = self.position_sampler(**position_sampler_args)

            if self.problem_type == 'SR':
                label_batch[iimage] = SR_label

            # sample bitpatterns
            if self.item_sampler is None:
                items_list, SD_label = self.sample_bitpatterns(SD_label=label_batch[iimage] if (self.problem_type == 'SD') else None,
                                                     SD_portion=self.SD_portion)
            else:
                items_list, SD_label = self.item_sampler(**item_sampler_args)

            if self.problem_type == 'SD':
                label_batch[iimage] = SD_label
	
            if self.mix_rules == True:
                label_batch[iimage] = SD_label*SR_label + (1-SD_label)*(1-SR_label)

            # perforate
            if self.perforate_type is not None:
                overlaps, new_tracker_mask, _ = self.perforate_type(positions_list, items_list,
                                                                    self.box_extent, self.item_size, self.num_item_pixel_values,
                                                                    self.perforate_mask, self.tracker_mask.copy())
                if (overlaps > 0.) & self.perforate:
                    continue
                self.tracker_mask = new_tracker_mask


            # print('label_batch:', label_batch[iimage])

            # render
            image = self.render(items_list, positions_list, label_batch[iimage], display=self.display)
            target_output[iimage, 0, 0, label_batch[iimage]] = 1
            # print('target_output:', target_output)
            input_data[iimage, :, :, :] = image
            # print('input_data:', input_data)
            iimage+=1
            if self.display:
                # print(target_output[iimage-1,0,0,:])
                positions_list_batch.append(positions_list)

            items_list_batch.append(items_list)

        return input_data, target_output, positions_list_batch, items_list_batch


    def get_tracker(self):
        if self.tracker_mask is None:
            return None
        else:
            return self.tracker_mask.copy()


    ############### DRAW
    def render(self, items_list, positions_list, SR_label, display=False):

        if len(items_list) != len(positions_list):
            raise ValueError('Should provide the same number of hard-coded items and positions')
        #   if (len(items_list)>2) | (len(positions_list)>2):
        #       raise ValueError('#items should not exceed 2 (not yet implemented)')

        mask = np.zeros(shape=(self.raw_input_size[0],self.raw_input_size[1]))
        image = np.zeros(shape=tuple(self.raw_input_size))

        for i, (position,item) in enumerate(zip(positions_list, items_list)):
            square_size = items_list[i].shape
            image[position[0]:position[0] + square_size[0], position[1]:position[1] + square_size[1], :] = items_list[i].copy()
            mask[position[0]:position[0] + square_size[0], position[1]:position[1] + square_size[1]] += 1
            if np.any(mask > 1):
                raise ValueError('Spatially overlapping items found.')
        if display:
            label = 'Horizontal' if SR_label == 0 else 'Vertical'
            plt.imshow(np.squeeze(image), interpolation='none')
            plt.title('SR_label: ' + label)
            plt.colorbar()
            plt.show()
            plt.clf()
        return image


    ############### SAMPLE
    def sample_positions(self, SR_label=None, SR_portion=1, SR_type = 'average_orientation'):
        if SR_label is None:
            SR_label = np.random.randint(low=0, high=2)

        positions_list = []
        running_orientation = 0
        running_displacement = np.array([0,0])
        for pp in range(self.num_items):
            position_flag = 0
            while position_flag == 0:
                position_flag = 1
                if not self.easy:
                    new_position =  [np.random.randint(low=0, high=self.box_extent[0] - (self.item_size[0] - 1)),
                    np.random.randint(low=0, high=self.box_extent[1] - (self.item_size[1] - 1))]
                else:
                    new_position =  [np.random.randint(low=0, high=self.raw_input_size[0]/20),
                                     np.random.randint(low=0, high=self.raw_input_size[1]/20)]
                    # print('new_position divided by 20:', new_position)
                    new_position[0] *= 20
                    new_position[1] *= 20
                for old_position in positions_list:
                    # print('positions list in the loop:',positions_list)
                    # print('old position:', old_position)
                    # print('new position:', new_position)
                    # print('SR Label:', SR_label)
                    # print('SR Portion:', SR_portion)
                    # print('Item Size:', self.item_size)
                    # print('SR Type:', SR_type)
                    position_viability = utility.check_position_viability(new_position, old_position, SR_label, SR_portion, self.item_size, SR_type)
                    position_flag *= position_viability
            positions_list.append(new_position)
            # print('positions list:', positions_list)
        if SR_type == 'average_orientation' or SR_type == 'average_displacement':
            for pos_1 in positions_list:
                for pos_2 in positions_list:
                    if np.all(pos_1 == pos_2):
                        continue
                    y = np.abs(np.array(pos_1)[0] - np.array(pos_2)[0])
                    x = np.abs(np.array(pos_1)[1] - np.array(pos_2)[1])
                    running_displacement += np.array([y,x])
                    running_orientation += np.arctan(y/x)

            #         print('running displacement:',running_displacement)
            #         print('running orientation inside:', running_orientation)
            # print('running orientation outside:', running_orientation)
            running_displacement = running_displacement / self.num_items*(self.num_items - 1)
            running_orientation = running_orientation / self.num_items*(self.num_items - 1)
            # print('running orientation division:', running_orientation)
            if SR_type == 'average_orientation':
                if np.abs(running_orientation) >= np.pi/4:
                    new_label = 1
                else:
                    new_label = 0
            elif SR_type == 'average_displacement':
                if running_displacement[0] > SR_portion*running_displacement[1]:
                    new_label = 1
                else:
                    new_label = 0
        else:
            raise ValueError('SR_type should be average_orientation or average_displacement.')
        #elif SR_type == 'all':
        #    new_label = SR_label
        return positions_list, new_label


    def sample_bitpatterns(self, SD_label=None, SD_portion=0):
        if (SD_portion>=1) | (SD_portion<0):
            raise ValueError('0<=SD_portion<1')

        # print('self.item_size:', self.item_size)
        base_item_sign_exponents = np.random.randint(low=0, high=2, size=self.item_size)
        # print("base_item_sign_exponents:", base_item_sign_exponents)
        base_item_values = np.random.randint(low=1, high=self.num_item_pixel_values + 1, size=self.item_size)
        # print("base_item_values:", base_item_values)
        base_item = np.power(-1, base_item_sign_exponents) * base_item_values
        # print("base_item:", base_item)

        items_list = [base_item.copy()]*self.num_items
        # print("items_list:", items_list)

        if SD_label is None:
            SD_label = np.random.randint(low=0, high=2)

        if SD_label == 0: # Different
            num_different = self.num_items - 1 #np.random.randint(low=1, high=self.num_items)
            for dd in range(num_different):
                item = items_list[dd].copy()
                item_flat = np.reshape(item, (-1, 1, self.item_size[2]))
                order = list(range(len(item_flat)))
                random.shuffle(order)
                min_num_diff_pixels = int(np.floor(SD_portion * np.float(reduce(mul, self.item_size)))) + 1
                for i in range(0,min_num_diff_pixels):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=True)
                for i in range(min_num_diff_pixels, item_flat.shape[0]):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=False)
                items_list[dd] = np.reshape(item_flat, tuple(self.item_size))
            for ss in range(num_different, self.num_items):
                item = items_list[ss].copy()
                item_flat = np.reshape(item, (-1, 1, self.item_size[2]))
                order= list(range(len(item_flat)))
                random.shuffle(order)
                max_num_diff_pixels = int(np.floor(SD_portion * np.float(reduce(mul, self.item_size))))
                for i in range(0,max_num_diff_pixels):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=False)

        elif SD_label == 1: # SAME
            num_different = np.random.randint(low=0, high=self.num_items-1)
            for dd in range(num_different):
                item = items_list[dd].copy()
                item_flat = np.reshape(item, (-1, 1, self.item_size[2]))
                order = list(range(len(item_flat)))
                random.shuffle(order)
                min_num_diff_pixels = int(np.floor(SD_portion * np.float(reduce(mul, self.item_size)))) + 1
                for i in range(0, min_num_diff_pixels):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=True)
                for i in range(min_num_diff_pixels, item_flat.shape[0]):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=False)
                items_list[dd] = np.reshape(item_flat, tuple(self.item_size))
            for ss in range(num_different, self.num_items):
                item = items_list[ss].copy()
                item_flat = np.reshape(item, (-1, 1, self.item_size[2]))
                order = list(range(len(item_flat)))
                random.shuffle(order)
                max_num_diff_pixels = int(np.floor(SD_portion * np.float(reduce(mul, self.item_size))))
                for i in range(0, max_num_diff_pixels):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=False)
        else:
            raise ValueError('SD_label should be 0 or 1 or None')

        # print("items_list:", items_list)
        # print("SD Label:", SD_label)
        return items_list, SD_label

    def resample_pixel(self, x, force_different):
        resampled = np.random.randint(low=-self.num_item_pixel_values, high=self.num_item_pixel_values, size=[1,1,self.item_size[2]])
        for ich in range(self.item_size[2]):
            if resampled[0,0,ich] >= 0:
                resampled[0,0,ich] += 1
        while (force_different) & (np.all(resampled == x)):
            resampled = self.resample_pixel(x, force_different)
        return resampled


############### PERFORATION
def perf_by_conditional_pos(positions_list, items_list, box_extent, item_size, num_values, perforate_mask, tracker_mask):
    mask_height = box_extent[0] - item_size[0] + 1
    legal_width = box_extent[1] - item_size[1] + 1
    vertical_ceiling_index = mask_height-1
    horizontal_center_index = box_extent[1] - item_size[1]
    if perforate_mask is None:
        perforate_mask = np.zeros((mask_height,legal_width*2-1))
        perforate_mask[vertical_ceiling_index-item_size[0]:,
                       horizontal_center_index-item_size[1]:horizontal_center_index+item_size[1]+1] = 1
    if tracker_mask is None:
        tracker_mask = np.zeros_like(perforate_mask)
    if (perforate_mask.shape[0]!=mask_height)|(perforate_mask.shape[1]!=legal_width*2-1):
        ValueError('mask shape does not match image size. should be [box_extent[0] - item_size[0] + 1, 2*box_extent[1] - 2*item_size[1] - 1]')
    if (positions_list is None) and (items_list is None):
        return 0, tracker_mask, perforate_mask

    displacement = [ -(positions_list[1][0] - positions_list[0][0]), positions_list[1][1] - positions_list[0][1]]
    if displacement[0] < 0:
        displacement = [-displacement[0], -displacement[1]] # Displacement wrt. to the lower item (up = positive)
    position_in_mask = [vertical_ceiling_index - displacement[0], horizontal_center_index + displacement[1]]
    overlaps = perforate_mask[position_in_mask[0], position_in_mask[1]]
    updated_tracker_mask = tracker_mask.copy()
    updated_tracker_mask[position_in_mask[0], position_in_mask[1]] += 1

    return overlaps, updated_tracker_mask, perforate_mask # an integet (#overlaps of this arrangement), a new mask reflecting the new example


def sample_positions_by_conditional_pos(nominal_relative_coordinate, box_extent, item_size, SR_portion):
    # re-create mask size
    mask_height = box_extent[0] - item_size[0] + 1
    mask_width = (box_extent[1] - item_size[1] + 1)*2 - 1
    vertical_ceiling_index = mask_height - 1
    horizontal_center_index = box_extent[1] - item_size[1]

    # convert to real coordinate (up-down is flipped here), representing the actual coordinate shift relative to reference.
    displacement = list(nominal_relative_coordinate) ###################### SHOULD COPY A LIST LIKE THIS, OR copy.copy(mylist) OR mylist[:]
    displacement[0] = vertical_ceiling_index - nominal_relative_coordinate[0]
    displacement[1] = nominal_relative_coordinate[1] - horizontal_center_index

    # set the legal range of reference position (inclusive)
    vertical_pos_range = [displacement[0], vertical_ceiling_index]
    horizontal_pos_range = [-np.minimum(0,displacement[1]), horizontal_center_index-np.maximum(0,displacement[1])]

    # sample
    y_pos_1 = np.random.randint(low=vertical_pos_range[0],high=vertical_pos_range[1]+1)
    x_pos_1 = np.random.randint(low=horizontal_pos_range[0],high=horizontal_pos_range[1]+1)

    positions_list = [[y_pos_1, x_pos_1],
                      [y_pos_1-displacement[0], x_pos_1+displacement[1]]]

    # determine SR label
    if (np.abs(displacement[1]) <= (SR_portion * np.float(np.abs(displacement[0])))):
        SR_label = 0
    else:
        SR_label = 1
    return positions_list, SR_label


def perf_by_marginal_pos(positions_list, items_list, box_extent, item_size, num_values, perforate_mask, tracker_mask):
    if perforate_mask is None:
        perforate_mask = np.zeros((box_extent[0]-item_size[0]+1,box_extent[1]-item_size[1]+1))
    if tracker_mask is None:
        tracker_mask = np.zeros_like(perforate_mask)
    if (perforate_mask.shape[0]!=box_extent[0])|(perforate_mask.shape[1]!=box_extent[1]):
        ValueError('mask shape does not match image size. should be [height-item_size[0]+1, width-item_size[1]+1]')
    if (positions_list is None) and (items_list is None):
        return 0, tracker_mask, perforate_mask

    overlaps = np.maximum(perforate_mask[positions_list[0][0], positions_list[0][1]], perforate_mask[positions_list[1][0], positions_list[1][1]])
    updated_tracker_mask = tracker_mask.copy()
    updated_tracker_mask[positions_list[0][0], positions_list[0][1]] += 1
    updated_tracker_mask[positions_list[1][0], positions_list[1][1]] += 1

    return overlaps, updated_tracker_mask, perforate_mask # a list of two numbers (#overlaps of each item's position), a new mask reflecting the new example


def sample_positions_by_marginal_pos(marginal_pos, box_extent, item_size, SR_portion):
    # SECOND BITPATTERN POSITION
    y_pos_2 = np.random.randint(low=0, high=box_extent[0] - (item_size[0] - 1))
    x_pos_2 = np.random.randint(low=0, high=box_extent[1] - (item_size[1] - 1))

    # RESAMPLE IF NECESSARY
    while (np.abs(marginal_pos[0] - y_pos_2) <= item_size[0]) & (np.abs(marginal_pos[1] - x_pos_2) <= item_size[1]):
        y_pos_2 = np.random.randint(low=0, high=box_extent[0] - (item_size[0] - 1))

    if (np.abs(x_pos_2 - marginal_pos[1]) <= (SR_portion * np.float(np.abs(y_pos_2 - marginal_pos[0])))):
        SR_label = 0
    else:
        SR_label = 1

    positions_list = [marginal_pos,[y_pos_2, x_pos_2]]
    return positions_list, SR_label


def perf_by_joint_pos(positions_list, items_list, box_extent, item_size, num_values, perforate_mask, tracker_mask):
    if perforate_mask is None:
        perforate_mask = np.zeros((box_extent[0]-item_size[0]+1,box_extent[1]-item_size[1]+1,box_extent[0]-item_size[0]+1,box_extent[1]-item_size[1]+1))
        for y in range(perforate_mask.shape[0]):
            for x in range(perforate_mask.shape[1]):
                perforate_mask[y,x,y-item_size[0]:y+item_size[0]+1,x-item_size[1]:x+item_size[1]+1] = 1
                perforate_mask[y-item_size[0]:y+item_size[0]+1,x-item_size[1]:x+item_size[1]+1,y,x] = 1
    if tracker_mask is None:
        tracker_mask = np.zeros_like(perforate_mask)
    if len(perforate_mask.shape) != 4:
        ValueError('mask shape should be 2x(num_items) dimensional')
    if (perforate_mask.shape[0]!=box_extent[0])|(perforate_mask.shape[1]!=box_extent[1])|(perforate_mask.shape[2]!=box_extent[0])|(perforate_mask.shape[3]!=box_extent[1]):
        ValueError('mask shape does not match image size. should be [height, width, height, width]')
    if (positions_list is None) and (items_list is None):
        return 0, tracker_mask, perforate_mask

    overlaps = perforate_mask[positions_list[0][0], positions_list[0][1], positions_list[1][0], positions_list[1][1]]
    updated_tracker_mask = tracker_mask.copy()
    updated_tracker_mask[positions_list[0][0], positions_list[0][1], positions_list[1][0], positions_list[1][1]] += 1
    updated_tracker_mask[positions_list[1][0], positions_list[1][1], positions_list[0][0], positions_list[0][1]] += 1

    return overlaps, updated_tracker_mask, perforate_mask # a list of two numbers (#overlaps of each item's position), a new mask reflecting the new example


def sample_positions_by_joint_pos(joint_pos, box_extent, item_size, SR_portion):

    if (np.abs(joint_pos[3] - joint_pos[1]) <= (SR_portion * np.float(np.abs(joint_pos[2] - joint_pos[0])))):
        SR_label = 0
    else:
        SR_label = 1

    positions_list = [[joint_pos[0],joint_pos[1]],[joint_pos[2],joint_pos[3]]]
    return positions_list, SR_label


#### TODO: Not implemented yet.
def perf_by_marginal_val(positions_list, items_list, box_extent, item_size, num_values, perforate_mask, tracker_mask):
    if perforate_mask is None:
        perforate_mask = None #NOT IMPLEMENTED
    if tracker_mask is None:
        tracker_mask = np.zeros_like(perforate_mask)
    if len(perforate_mask.shape) != 3:
        ValueError('mask shape should be 3 dimensional')
    if (perforate_mask.shape[0]!=item_size[0])|(perforate_mask.shape[1]!=item_size[1])|(perforate_mask.shape[2]!=2*num_values):
        ValueError('mask shape does not match image size. should be [height, width, height, width]')
    if (positions_list is None) and (items_list is None):
        return 0, tracker_mask, perforate_mask
    return



# For testing
# if __name__ == '__main__':
#     batch_size = 5
#     raw_input_size = [100,100,1]
#     item_size = [5,5]
#     box_extent = [100,100]
#     num_item_pixel_values = 1
#     problem_type = 'SR'
#     # perforation
#     perforate_type = None
#     #perforate_type = perf_by_conditional_pos
#     #_, _, tracker_mask = perforate_type(None, None, box_extent, item_size, num_item_pixel_values, None, None)
#     #perforate_mask = np.random.randint(0,2,size=tracker_mask.shape)
#     #perforate = True
#
#     generator = psvrt(raw_input_size, batch_size)
#     generator.initialize_vars(problem_type, item_size, box_extent,
#                         num_items=3, num_item_pixel_values=num_item_pixel_values, SD_portion=0, SR_portion=1, SR_type = 'average_orientation',
#                         perforate_type=perforate_type,
#                         display=True)
#
#     input_data, target_output, positions_list_batch, items_list_batch = generator.single_batch()
#     updated_mask = generator.get_tracker()
#
#     print(target_output)
#     plt.imshow(updated_mask, interpolation='none')
#     plt.show()
#     plt.imshow(perforate_mask, interpolation='none')
#     plt.show()

"""
        if SD_label == 0: # Different
            num_different = self.num_items - 1 #np.random.randint(low=1, high=self.num_items)
            for dd in range(num_different):
                item = items_list[dd].copy()
                item_flat = np.reshape(item, (-1, 1, self.item_size[2]))
                order = range(len(item_flat))
                random.shuffle(order)
                min_num_diff_pixels = int(np.floor(SD_portion * np.float(reduce(mul, self.item_size)))) + 1
                for i in range(0,min_num_diff_pixels):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=True)
                for i in range(min_num_diff_pixels, item_flat.shape[0]):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=False)
                items_list[dd] = np.reshape(item_flat, tuple(self.item_size))
            for ss in range(num_different, self.num_items):
                item = items_list[ss].copy()
                item_flat = np.reshape(item, (-1, 1, self.item_size[2]))
                order= range(len(item_flat))
                random.shuffle(order)
                max_num_diff_pixels = int(np.floor(SD_portion * np.float(reduce(mul, self.item_size))))
                for i in range(0,max_num_diff_pixels):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=False)

        elif SD_label == 1: # SAME
            num_different = np.random.randint(low=0, high=self.num_items-1)
            for dd in range(num_different):
                item = items_list[dd].copy()
                item_flat = np.reshape(item, (-1, 1, self.item_size[2]))
                order = range(len(item_flat))
                random.shuffle(order)
                min_num_diff_pixels = int(np.floor(SD_portion * np.float(reduce(mul, self.item_size)))) + 1
                for i in range(0, min_num_diff_pixels):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=True)
                for i in range(min_num_diff_pixels, item_flat.shape[0]):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=False)
                items_list[dd] = np.reshape(item_flat, tuple(self.item_size))
            for ss in range(num_different, self.num_items):
                item = items_list[ss].copy()
                item_flat = np.reshape(item, (-1, 1, self.item_size[2]))
                order = range(len(item_flat))
                random.shuffle(order)
                max_num_diff_pixels = int(np.floor(SD_portion * np.float(reduce(mul, self.item_size))))
                for i in range(0, max_num_diff_pixels):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=False)
        else:
            raise ValueError('SD_label should be 0 or 1 or None')

"""
"""
        if SD_label == 0: # Different
            num_different = np.random.randint(low=1, high=self.num_items)
            for dd in range(num_different):
                item = items_list[dd].copy()
                item_flat = np.reshape(item, (-1,1,self.item_size[2]))
                order = range(item_flat.shape[0])
                random.shuffle(order)
                min_num_diff_pixels = int(np.floor(SD_portion * np.float(reduce(mul, self.item_size)))) + 1
                for i in range(0,min_num_diff_pixels):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=True)
                for i in range(min_num_diff_pixels,item_flat.shape[0]):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=False)
                items_list[dd] = np.reshape(item_flat, tuple(self.item_size)).copy()

        elif SD_label == 1: # SAME
            for dd in range(self.num_items):
                item = items_list[dd].copy()
                item_flat = np.reshape(item, (-1, 1, self.item_size[2]))
                order = range(item_flat.shape[0])
                random.shuffle(order)
                max_num_diff_pixels = int(np.floor(SD_portion * np.float(reduce(mul, self.item_size))))
                for i in range(0, max_num_diff_pixels):
                    item_flat[order[i],0,:] = self.resample_pixel(item_flat[order[i],0,:], force_different=False)
                items_list[dd] = np.reshape(item_flat, tuple(self.item_size)).copy()
        else:
            raise ValueError('SD_label should be 0 or 1 or None')
"""
