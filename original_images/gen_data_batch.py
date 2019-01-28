import psvrt
import numpy as np
from utility import rgb2grey, sep_boxes
import matplotlib.pyplot as plt

#raw_input_size = [6,10,3] # Size of whole image
batch_size = 2

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
	left_mask = np.reshape(left_mask, [batch_size,img_shape[0] * img_shape[1]])
	right_mask = np.reshape(right_mask, [batch_size, img_shape[0] * img_shape[1]])


	return train_data, mask_labels, left_mask, right_mask, ret_lbls


def generate_batch_1(batch_size, img_shape, itm_size, n_itms, prob_type = 'SD'):
	np.random.seed(100)
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


# train_data, mask_labels, ret_lbls = generate_batch_1(2, (2,2,3), (1,1), 1)
# print(train_data.shape)
# train_data = np.reshape(train_data, [2,2,3])
# print(mask_labels.shape)
# mask_labels = np.reshape(mask_labels, [])

# mask = np.array([0.0, 0.09765625, 0.1484375, 0.2578125, 0.32421875, 0.3828125, 0.34375, 0.43359375, 0.41015625, 0.46484375, 0.5234375, 0.50390625, 0.515625, 0.53515625, 0.58203125, 0.62109375, 0.59765625, 0.58203125, 0.546875, 0.60546875, 0.6015625, 0.62109375, 0.60546875, 0.63671875, 0.70703125, 0.70703125, 0.70703125, 0.70703125, 0.72265625, 0.6953125, 0.69140625, 0.73046875, 0.74609375, 0.7265625, 0.6484375, 0.73046875, 0.7578125, 0.66796875, 0.75390625, 0.72265625, 0.76171875, 0.75, 0.74609375, 0.796875, 0.7265625, 0.71875, 0.796875, 0.796875, 0.74609375, 0.77734375, 0.734375, 0.734375, 0.79296875, 0.796875, 0.796875, 0.74609375, 0.78515625, 0.81640625, 0.76953125, 0.81640625, 0.78515625, 0.7890625, 0.8203125, 0.828125, 0.8046875, 0.8125, 0.78125, 0.796875, 0.80859375, 0.82421875, 0.76171875, 0.80078125, 0.77734375, 0.8125, 0.80859375, 0.86328125, 0.8203125, 0.8125, 0.796875, 0.8125, 0.828125, 0.796875, 0.85546875, 0.80859375, 0.77734375, 0.79296875, 0.828125, 0.8203125, 0.7734375, 0.8203125, 0.82421875, 0.8203125, 0.8125, 0.84375, 0.83984375, 0.81640625, 0.8203125, 0.88671875, 0.87109375, 0.8359375, 0.84765625, 0.8359375, 0.88671875, 0.8203125, 0.8671875, 0.83984375, 0.859375, 0.8671875, 0.86328125, 0.83203125, 0.8671875, 0.8203125, 0.83203125, 0.84375, 0.84375, 0.875, 0.85546875, 0.8515625, 0.8671875, 0.8671875, 0.828125, 0.81640625, 0.83984375, 0.86328125, 0.84765625, 0.84765625, 0.875, 0.8828125, 0.85546875, 0.89453125, 0.8984375, 0.86328125, 0.859375, 0.87890625, 0.8515625, 0.8828125, 0.84375, 0.85546875, 0.88671875, 0.87109375, 0.8828125, 0.83203125, 0.86328125, 0.87109375, 0.890625, 0.8984375, 0.87109375, 0.8515625, 0.84765625, 0.8828125, 0.84765625, 0.8671875, 0.83984375, 0.85546875, 0.88671875, 0.8515625, 0.90625, 0.84375, 0.87109375, 0.87109375])
# sd = np.array([0.46484375, 0.5, 0.5078125, 0.5, 0.5, 0.4765625, 0.48828125, 0.49609375, 0.546875, 0.546875, 0.515625, 0.484375, 0.484375, 0.5859375, 0.52734375, 0.4921875, 0.53515625, 0.5, 0.4765625, 0.50390625, 0.53125, 0.5, 0.4375, 0.48046875, 0.48828125, 0.4609375, 0.46875, 0.47265625, 0.4375, 0.54296875, 0.51171875, 0.453125, 0.50390625, 0.484375, 0.48828125, 0.48046875, 0.578125, 0.53125, 0.47265625, 0.47265625, 0.5078125, 0.5234375, 0.48046875, 0.46484375, 0.484375, 0.52734375, 0.45703125, 0.5234375, 0.52734375, 0.49609375, 0.48828125, 0.59375, 0.50390625, 0.453125, 0.50390625, 0.5, 0.51953125, 0.49609375, 0.51953125, 0.51953125, 0.484375, 0.5546875, 0.5078125, 0.515625, 0.4609375, 0.46484375, 0.50390625, 0.515625, 0.49609375, 0.484375, 0.46484375, 0.4765625, 0.53125, 0.51953125, 0.51171875, 0.515625, 0.58984375, 0.5234375, 0.4609375, 0.4921875, 0.51171875, 0.515625, 0.5, 0.49609375, 0.5078125, 0.515625, 0.49609375, 0.48828125, 0.515625, 0.5078125, 0.53125, 0.515625, 0.48828125, 0.46875, 0.5, 0.50390625, 0.51953125, 0.5, 0.48046875, 0.5, 0.53125, 0.55859375, 0.48046875, 0.5, 0.48828125, 0.48046875, 0.58203125, 0.4765625, 0.48828125, 0.5, 0.47265625, 0.53515625, 0.50390625, 0.484375, 0.4609375, 0.484375, 0.515625, 0.55859375, 0.51171875, 0.5, 0.45703125, 0.50390625, 0.5, 0.4765625, 0.48828125, 0.48046875, 0.5390625, 0.453125, 0.6484375, 0.6953125, 0.6875, 0.64453125, 0.69921875, 0.69921875, 0.7578125, 0.7265625, 0.734375, 0.74609375, 0.71484375, 0.69140625, 0.70703125, 0.7265625, 0.80078125, 0.7734375, 0.74609375, 0.7421875, 0.73046875, 0.7265625, 0.78515625, 0.75390625, 0.75, 0.7890625, 0.78125, 0.78515625, 0.7734375, 0.7578125, 0.76953125, 0.75, 0.8046875, 0.8046875])
# sr = np.array([0.8359375, 0.8515625, 0.8203125, 0.83984375, 0.828125, 0.8359375, 0.8515625, 0.8515625, 0.84765625, 0.83984375, 0.84765625, 0.83203125, 0.8828125, 0.87109375, 0.83984375, 0.85546875, 0.84375, 0.8515625, 0.8203125, 0.8671875, 0.8359375, 0.890625, 0.859375, 0.82421875, 0.81640625, 0.8671875, 0.85546875, 0.86328125, 0.85546875, 0.85546875, 0.8203125, 0.8515625, 0.89453125, 0.84375, 0.83984375, 0.88671875, 0.82421875, 0.86328125, 0.85546875, 0.89453125, 0.87109375, 0.8828125, 0.84765625, 0.8203125, 0.88671875, 0.84765625, 0.85546875, 0.86328125, 0.89453125, 0.83203125, 0.8671875, 0.84765625, 0.90625, 0.8515625, 0.83984375, 0.87109375, 0.8828125, 0.84375, 0.8828125, 0.87109375, 0.8828125, 0.8515625, 0.87890625, 0.859375, 0.86328125, 0.87890625, 0.88671875, 0.86328125, 0.8359375, 0.85546875, 0.8515625, 0.86328125, 0.83203125, 0.83984375, 0.859375, 0.8359375, 0.84765625, 0.828125, 0.859375, 0.84375, 0.78515625, 0.8671875, 0.82421875, 0.8515625, 0.859375, 0.859375, 0.875, 0.859375, 0.87890625, 0.83203125, 0.87890625, 0.83984375, 0.83984375, 0.8515625, 0.8828125, 0.859375, 0.87109375, 0.84765625, 0.87890625, 0.859375, 0.8671875, 0.87109375, 0.890625, 0.87109375, 0.8828125, 0.8671875, 0.83984375, 0.84765625, 0.83984375, 0.859375, 0.87890625, 0.8671875, 0.859375, 0.84375, 0.84375, 0.87890625, 0.8515625, 0.8671875, 0.85546875, 0.84375, 0.87109375, 0.89453125, 0.85546875, 0.8515625, 0.8671875, 0.890625, 0.8828125, 0.86328125, 0.85546875, 0.8359375, 0.82421875, 0.875, 0.8828125, 0.87109375, 0.8828125, 0.8828125, 0.87890625, 0.87109375, 0.8984375, 0.84765625, 0.8828125, 0.8203125, 0.81640625, 0.859375, 0.87890625, 0.87109375, 0.8359375, 0.85546875, 0.87109375, 0.8828125, 0.84765625, 0.8359375, 0.8359375, 0.84765625, 0.86328125, 0.8671875, 0.875, 0.8515625, 0.90234375, 0.890625])
#
# train_size = np.arange(1, len(mask) + 1)
#
#
#
# plt.figure(1,figsize=(7,5))
# plt.plot(train_size,mask)
# plt.plot(train_size, sd)
# plt.plot(train_size,sr)
# plt.xlabel('Training Size')
# plt.ylabel("Accuracy")
# plt.title("SD vs SR vs Mask")
# plt.grid(True)
# plt.legend(['(60x60 Image, 5x5 Patch Size)'])
# plt.style.use(['classic'])
# plt.show()
