import psvrt
import numpy as np
from utility import rgb2grey, sep_boxes
import matplotlib.pyplot as plt

#raw_input_size = [6,10,3] # Size of whole image
batch_size = 2




def generate_batch_1(batch_size, img_shape, itm_size, n_itms, prob_type = 'SD'):
	# np.random.seed(100)
	"""

	This function is written to generate & return a batch of images and returns mask for the item inside it. Precisely, a single item. It's only written to test a network.
	Please use generate_batch instead if there are 2 or more items in an image.

	:param batch_size:
	:param img_shape:
	:param itm_size:
	:param n_itms:
	:param prob_type:
	:return:
	"""
	# Change `problem_type` to SR for spatial relation labels
	data_parameters = {'problem_type': prob_type,
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
	mask_labels = rgb2grey(np.array([items[0] for items in all_items]))

	ret_lbls = np.zeros([batch_size, 2])
	indices_1 = np.where(labels_temp == 1)
	indices_0 = np.where(labels_temp == 0)

	ret_lbls[indices_0] = [1,0]
	ret_lbls[indices_1] = [0,1]
	mask_labels = np.reshape(mask_labels, [mask_labels.shape[0], mask_labels.shape[1] * mask_labels.shape[2]])
	# left_mask, right_mask = sep_boxes(img_shape, itm_size, n_itms, mask_labels, labels_temp)
	# left_mask = np.reshape(left_mask, [batch_size,img_shape[0] * img_shape[1]])
	# right_mask = np.reshape(right_mask, [batch_size, img_shape[0] * img_shape[1]])


	return train_data, mask_labels, ret_lbls

def generate_batch(batch_size, img_shape, itm_size, n_itms, prob_type = 'SD'):
	"""

	This function is written to generate & return a batch of images and returns mask for the items inside it. Precisely, 2 items(left & right)
	:param batch_size:
	:param img_shape:
	:param itm_size:
	:param n_itms:
	:param prob_type:
	:return:
	"""
	# Change `problem_type` to SR for spatial relation labels
	data_parameters = {'problem_type': prob_type,
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
	mask_labels = rgb2grey(np.array([items[0] for items in all_items]))

	ret_lbls = np.zeros([batch_size, 2])
	indices_1 = np.where(labels_temp == 1)
	indices_0 = np.where(labels_temp == 0)

	ret_lbls[indices_0] = [1,0]
	ret_lbls[indices_1] = [0,1]

	left_mask, right_mask = sep_boxes(img_shape, itm_size, n_itms, mask_labels, labels_temp)

	"""This part of the code is written to add 2 dimensions for every pixel, whether that pixel is 0 or 1."""
	# ret_lmask = np.zeros([batch_size, img_shape[0] * img_shape[1], 2])
	# ret_rmask = np.zeros([batch_size, img_shape[0] * img_shape[1], 2])

	left_mask = np.reshape(left_mask, [batch_size,img_shape[0] * img_shape[1]])
	# indices_lmask_1 = np.where(left_mask == 1)
	# indices_lmask_0 = np.where(left_mask == 0)
	#
	#
	# ret_lmask[indices_lmask_0]  = [1.,0.]
	# ret_lmask[indices_lmask_1]  = [0.,1.]

	# print(ret_lmask)
	# left_mask[0, indices_lmask_1] = [0.,1.]


	right_mask = np.reshape(right_mask, [batch_size, img_shape[0] * img_shape[1]])

	# indices_rmask_1 = np.where(right_mask == 1)
	# indices_rmask_0 = np.where(right_mask == 0)
	#
	# ret_rmask[indices_rmask_0]  = [1.,0.]
	# ret_rmask[indices_rmask_1]  = [0.,1.]

	# print(left_mask)
	# print(right_mask)

	return train_data, mask_labels, left_mask, right_mask, ret_lbls


# train_data, mask_labels, left_mask, right_mask, ret_lbls = generate_batch(2, (6,6,3), (2,2), 2)
# print(left_mask, right_mask)