import psvrt
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

raw_input_size = [4,4,3] # Size of whole image
batch_size = 3

# Change `problem_type` to SR for spatial relation labels
data_parameters = {'problem_type': 'SD',
		   'item_size': [2,2],
		   'box_extent': [4,4],
		   'num_items': 1,
		   'num_item_pixel_values': 1,
		   'SD_portion': 0,
		   'SR_portion': 1,
            'patch_coords': False, #This parameter is introduced to generate patch coordinates and size in a csv file.
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
# print('all items', all_items)

labels_temp = np.squeeze(labels[:, 0, 0, 0])

# print(len(stimuli), len(all_items), len(labels), len(label_positions))


# if data_parameters['patch_coords'] == True:
#     lbl_coords = open('label_coords.csv', 'w')
# writer = csv.writer(lbl_coords)

count = 1
for img, lbl,items in zip(stimuli, labels_temp, all_items):
    # print("I'm going to do something")
    # img = np.squeeze(img[:,:,0])
    # print('img shape:', img.shape)

    if not os.path.exists('./SD' + str(count)):
        os.makedirs('./SD/' + str(count))
        os.makedirs('./SD/' + str(count) + '/img')
        os.makedirs('./SD/' + str(count) + '/labels')

    plt.imsave('./SD/' + str(count) + '/img/img.png', img)
    item_count = 1

    if data_parameters['mask'] == True:
        items = np.expand_dims(items, axis=0)

    print(items.shape)

    for item in items:
        plt.imsave('./SD/' + str(count) + '/labels/' + str(item_count) + '_' + str(int(lbl)) + '.png', np.squeeze(item[:,:,:]))
        item_count += 1

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