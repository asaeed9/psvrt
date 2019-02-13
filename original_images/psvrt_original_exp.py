from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2, os, math, time, sys
from datetime import timedelta
from sklearn.utils import shuffle
sys.path.append('../../original_images')
from gen_data_batch import generate_batch
from scipy import *

###Configuration
"""
Data Configurations/Paths
"""
img_dir_patch="./SD/predicted_patches"
img_dir_orig = "../../original_images/SD"

sd_model = 'SD/sd_model.ckpt'
# img_type = "original"
img_type = "patch"

##
# Convolutional Layer 1.
filter_size1 = 4          # Convolution filters are 4 x 4 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 2          # Convolution filters are 2 x 2 pixels.
num_filters2 = 32         # There are 32 of these filters.

# Convolutional Layer 3.
filter_size3 = 2          # Convolution filters are 2 x 2 pixels.
num_filters3 = 64         # There are 64 of these filters.

# Convolutional Layer 4.
filter_size4 = 2          # Convolution filters are 2 x 2 pixels.
num_filters4 = 128         # There are 128 of these filters.

# Fully-connected layer.
fc_size = 256             # Number of neurons in fully-connected layer.

# Number of colour channels for the images: 3 channel for RGB.
num_channels = 3

# Tuple with height and width of images used to reshape arrays.
img_shape = (10, 10, num_channels)
item_size = (2,2)
# Number of classes, one class for same and one for different image
num_classes = 2
nitems = 2

total_imgs = 1024
train_batch_size = 128


def restore_see_layer(orig, input_name, model_name=None, var_name=None):
	result = []
	with tf.Session('', tf.Graph()) as s:
		with s.graph.as_default():
			if ((model_name != None) and var_name != None):
				saver = tf.train.import_meta_graph(model_name + ".meta")
				saver.restore(s, model_name)
				fd = {input_name + ':0': orig}
				#                 print(fd.shape)
				for var in var_name:
					var = var + ":0"
					#                     result = 0

					result.append(s.run(var, feed_dict=fd))
	return result[0]


def test_mask_sd(batch_size, model_mask_sd):
	np.set_printoptions(suppress=True)
	sd_accuracy = 0.
	mask_accuracy = 0.
	sig_mask_accuracy = 0.

	test_set, mask_test_labels, _, _, test_labels = generate_batch(batch_size, img_shape, item_size, nitems)
	sd = restore_see_layer(orig=test_set, input_name='x', model_name=model_mask_sd, var_name=['Softmax'])
	sd_accuracy = np.sum(np.argmax(sd, axis=1) == np.argmax(test_labels, axis=1)) / batch_size
	print('Mask Test Accuracy:{0:>.4f}, SD Test Accuracy:{1:>.4f}'.format(mask_accuracy, sd_accuracy))

	return mask_accuracy, sd_accuracy, 0., mask_test_labels

def new_weights(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))
#     return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_bias(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,
				   num_input_channels,
				   filter_size,
				   num_filters,
				   use_pooling=True):

	shape = [filter_size, filter_size, num_input_channels, num_filters]
	weights = new_weights(shape)
	biases = new_bias(length=num_filters)

	layer = tf.nn.conv2d(input=input,
						 filter=weights,
						 strides=[1, 1, 1, 1],
						 padding='SAME')
	layer += biases

	if use_pooling:
		layer = tf.nn.max_pool(value=layer,
							   ksize=[1, 3, 3, 1],
							   strides=[1, 2, 2, 1],
							   padding='SAME')
	layer = tf.nn.relu(layer)

	return layer, weights


def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer, [-1, num_features])

	return layer_flat, num_features


def new_fc_layer(input,
				 num_inputs,
				 num_outputs,
				 use_relu=True):
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_bias(length=num_outputs)
	layer = tf.matmul(input, weights) + biases

	if use_relu:
		layer = tf.nn.relu(layer)

	return layer

x = tf.placeholder(tf.float32, shape=[None, img_shape[0]*img_shape[1]*num_channels], name='x')
x_image = tf.reshape(x, [-1, img_shape[0], img_shape[1], num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
x_image.shape, y_true

layer1_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                            num_input_channels=num_channels,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=True)

layer2_conv2, weights_conv2 =  new_conv_layer(input=layer1_conv1,
                                           num_input_channels=num_filters1,
                                           filter_size=filter_size2,
                                           num_filters=num_filters2,
                                           use_pooling=True)

layer3_conv3, weights_conv3 =  new_conv_layer(input=layer2_conv2,
                                           num_input_channels=num_filters2,
                                           filter_size=filter_size3,
                                           num_filters=num_filters3,
                                           use_pooling=True)

layer4_conv4, weights_conv4 =  new_conv_layer(input=layer3_conv3,
                                           num_input_channels=num_filters3,
                                           filter_size=filter_size4,
                                           num_filters=num_filters4,
                                           use_pooling=True)

layer_flat, num_features = flatten_layer(layer4_conv4)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)


layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=fc_size,
                         use_relu=False)

layer_fc3 = new_fc_layer(input=layer_fc2,
                         num_inputs=fc_size,
                         num_outputs=fc_size,
                         use_relu=False)

layer_fc4 = new_fc_layer(input=layer_fc3,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

# drop_out = tf.nn.dropout(layer_fc4, 0.5)

##Normalize the numbers(apply softmax!)

y_pred = tf.nn.softmax(layer_fc4)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc4,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

## some more performance measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def optimize(num_epochs, save_model=True, save_name="base_model", restore_model=False, restore_name=None):
	total_iterations = 0
	done_train_imgs = 0
	start_time = time.time()
	start_batch = 0
	end_batch = train_batch_size
	test_batch_size = 256
	plot_sd = []
	saver = tf.train.Saver()
	sum_accuracy = 0.0
	n = 1

	# to save the model
	for i in range(0, num_epochs):
		start_batch = 0
		end_batch = train_batch_size

		print("Epoch:", i + 1)

		if restore_model == True:
			if restore_name == None:
				print("No model file specified")
				return
			else:
				saver.restore(session, restore_name)

		sum_accuracy = 0.0
		n = 1
		while end_batch < total_imgs:
			train_data, mask_test_labels, left_mask, right_mask, labels = generate_batch(train_batch_size, img_shape,
																						 item_size, nitems)

			if not len(train_data) and not len(labels):
				print("All images have been processed.")
				break;

			# x_batch, y_true_batch = next_batch(len(train), train, labels)
			feed_dict_train = {x: train_data,
							   y_true: labels}

			session.run(optimizer, feed_dict=feed_dict_train)

			acc, co = session.run([accuracy, cost], feed_dict=feed_dict_train)
			sum_accuracy += acc
			n += 1
			msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}, Loss: {2:>.4f}"
			print(msg.format(end_batch + 1, acc, co))

			start_batch += train_batch_size
			end_batch += train_batch_size

		if save_model == True:
			if save_name == None:
				print("No model specified, model not being saved")
				return
			else:
				save_path = saver.save(session, save_name)
				restore_model = True
				print("Model saved in file: %s" % save_name)

		s_time = time.time()
		mask_acc, sd_acc, _, _ = test_mask_sd(test_batch_size, save_name)
		plot_sd.append(sd_acc)
		e_time = time.time()
		time_dif = e_time - s_time
		print("Test Mask SD Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

	end_time = time.time()
	# Difference between start and end-times.
	time_dif = end_time - start_time

	# Print the time-usage.
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
	print(plot_sd)

if __name__ == "__main__":
	save_model = True
	save_name = sd_model
	restore_model = True
	restore_name = sd_model

	session = tf.Session()
	session.run(tf.global_variables_initializer())
	optimize(num_epochs=5, save_model=save_model, save_name=save_name, restore_model=restore_model,
			 restore_name=restore_name)
