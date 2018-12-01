import psvrt
import numpy as np

raw_input_size = [10,10,3] # Size of whole image
batch_size = 2

def generate_batch(batch_size, img_shape, itm_size, n_itms):
	# Change `problem_type` to SR for spatial relation labels
	data_parameters = {'problem_type': 'SD',
			   'item_size': [itm_size[0],itm_size[1]],
			   'box_extent': [img_shape[0], img_shape[1]],
			   'num_items': n_itms,
			   'num_item_pixel_values': 1,
			   'SD_portion': 0,
			   'SR_portion': 1,
				'full_size': 2, #whether ground label should be a full-size masked image or concatenated patches from the original image. 0 = merged patches, 1 = masked image, 2 = both
				'mask': True, #This parameter is introduced in order to generate label of the image as 1 where patch is and 0 else where.
			   'display': False}


	data_generator = psvrt.psvrt(raw_input_size=img_shape, batch_size=batch_size)  # Load generator
	data_generator.initialize_vars(**data_parameters)  # Initialize it with image parameters

	data = data_generator.single_batch() # Fetch a single batch

	stimuli = np.array(data[0]) # Images
	labels = np.array(data[1]) # Labels
	label_positions = np.array(data[2])
	all_items = np.asarray(data[3])

	labels_temp = np.array(np.squeeze(labels[:, 0, 0, 0]), np.int64)
	train_data = np.reshape(stimuli, [stimuli.shape[0], stimuli.shape[1] * stimuli.shape[2] * stimuli.shape[3]])

	ret_lbls = np.zeros([batch_size, 2])
	indices_1 = np.where(labels_temp == 1)
	indices_0 = np.where(labels_temp == 0)

	ret_lbls[indices_0] = [1,0]
	ret_lbls[indices_1] = [0,1]

	return train_data, ret_lbls

# train_data, ret_lbls = generate_batch(64)
# print(train_data.shape, ret_lbls.shape)