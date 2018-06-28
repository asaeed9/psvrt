import psvrt
import numpy as np
import matplotlib.pyplot as plt
import os

raw_input_size = [60,60,1] # Size of whole image
batch_size = 4000

# Change `problem_type` to SR for spatial relation labels
data_parameters = {'problem_type': 'SD',
		   'item_size': [20,20],
		   'box_extent': [60,60],
		   'num_items': 2,
		   'num_item_pixel_values': 1,
		   'SD_portion': 0,
		   'SR_portion': 1,
		   'display': False}

data_generator = psvrt.psvrt(raw_input_size=raw_input_size, batch_size=batch_size) # Load generator
data_generator.initialize_vars(**data_parameters) # Initialize it with image parameters

data = data_generator.single_batch() # Fetch a single batch

stimuli = data[0] # Images
labels = data[1] # Labels
all_items = np.asarray(data[3])

labels_temp = np.squeeze(labels[:, 0, 0, 0])

# print(len(stimuli), len(all_items), len(labels))

count = 1
for img, lbl,items in zip(stimuli, labels_temp, all_items):
	img = np.squeeze(img[:,:,0])

	if not os.path.exists('./SD' + str(count)):
		os.makedirs('./SD/' + str(count))
		os.makedirs('./SD/' + str(count) + '/img')
		os.makedirs('./SD/' + str(count) + '/labels')

	plt.imsave('./SD/' + str(count) + '/img/img.png', img)
	item_count = 1
	for item in items:
		# print(item.shape)
		# print(lbl)
		plt.imsave('./SD/' + str(count) + '/labels/' + str(item_count) + '_' + str(int(lbl)) + '.png', np.squeeze(item[:,:, 0]))
		item_count += 1

	# print('Image done.')

	count +=1

# plt.imshow(np.squeeze(stimuli[0,:,:,0]))
# plt.show()
# cv2.imwrite("cpImage.bmp", data[0][0])
