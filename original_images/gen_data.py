import psvrt
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

raw_input_size = [10,10,3] # Size of whole image
batch_size = 50000

# Change `problem_type` to SR for spatial relation labels
data_parameters = {'problem_type': 'SD',
		   'item_size': [2,2],
		   'box_extent': [10,10],
		   'num_items': 2,
		   'num_item_pixel_values': 1,
		   'SD_portion': 0,
		   'SR_portion': 1,
            'full_size': 2, #whether ground label should be a full-size masked image or concatenated patches from the original image. 0 = merged patches, 1 = masked image, 2 = both
			'mask': True, #This parameter is introduced in order to generate label of the image as 1 where patch is and 0 else where.
		   'display': False}

data_generator = psvrt.psvrt(raw_input_size=raw_input_size, batch_size=batch_size) # Load generator
data_generator.initialize_vars(**data_parameters) # Initialize it with image parameters

data = data_generator.single_batch() # Fetch a single batch

stimuli = data[0] # Images
labels = data[1] # Labels
label_positions = data[2]
all_items = np.asarray(data[3])

# print('stimuli:', stimuli)
# print('label positions: ', label_positions.shape)
labels_temp = np.array(np.squeeze(labels[:, 0, 0, 0]), np.int64)

# print(len(stimuli), len(all_items), len(labels), len(label_positions))

count = 1
for img, lbl,items in zip(stimuli, labels_temp, all_items):

    if not os.path.exists('./SD' + str(count)):
        os.makedirs('./SD/' + str(count))
        os.makedirs('./SD/' + str(count) + '/' + str(lbl))
        os.makedirs('./SD/' + str(count) + '/' + str(lbl) + '/img')
        os.makedirs('./SD/' + str(count) + '/' + str(lbl) + '/labels')

    plt.imsave('./SD/' + str(count) + '/' + str(lbl) +  '/img/img.png', img)
    item_count = 1

    if data_parameters['mask'] == True:
        items = np.expand_dims(items, axis=0)

    if data_parameters['full_size'] == 1:
        # for item in items:
        # plt.imsave('./SD/' + str(count) + '/' + str(lbl) + '/labels/' + str(item_count) + '_' + str(int(lbl)) + '.png', np.squeeze(item))
        # item_count += 1
        plt.imsave('./SD/' + str(count) + '/' + str(lbl) + '/labels/mask.png',np.squeeze(items))
    elif data_parameters['full_size'] == 0:
        plt.imsave('./SD/' + str(count) + '/' + str(lbl) + '/labels/merged_patch.png', np.squeeze(items))
    else:
        plt.imsave('./SD/' + str(count) + '/' + str(lbl) + '/labels/mask.png', np.squeeze(items)[0])
        plt.imsave('./SD/' + str(count) + '/' + str(lbl) + '/labels/merged_patch.png', np.squeeze(items)[1])


    # """Build a csv file which would have one line of label coordinates and size for every image"""
    # if data_parameters['patch_coords'] == True:
    #     lbl_row = ""
    #     lbl_row = str(count) +"," +str(data_parameters['item_size'][0])
    #
    #     for lbls in lbl_pos:
    #         x = lbls[0]
    #         y = lbls[1]
    #         lbl_row += "," + str(x)
    #         lbl_row += "," + str(y)
    #
    #     # print(lbl_row)
    #     lbl_coords.write(lbl_row)
    #     lbl_coords.write("\n")

    count +=1

# if data_parameters['patch_coords'] == True:
#     lbl_coords.close()