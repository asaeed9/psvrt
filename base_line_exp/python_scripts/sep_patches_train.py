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
from sklearn.preprocessing import normalize

###Configuration
"""
Data Configurations/Paths
"""
img_dir_patch="./SD/predicted_patches"
img_dir_orig = "./SD/original_images"
img_dir="../../original_images/SD"
model50_SD = 'SD/50kSD_Model.ckpt'
model50_left_mask = 'SD/50k_left_mask.ckpt'
model50_right_mask = 'SD/50k_right_mask.ckpt'

img_type = "original"
# img_type = "patch"

##
# Convolutional Layer 1.
filter_size0 = 16          # Convolution filters are 4 x 4 pixels.
num_filters0 = 64         # There are 16 of these filters.

filter_size1 = 8          # Convolution filters are 4 x 4 pixels.
num_filters1 = 64         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 8          # Convolution filters are 2 x 2 pixels.
num_filters2 = 32         # There are 32 of these filters.

filter_size3 = 8          # Convolution filters are 2 x 2 pixels.
num_filters3 = 32         # There are 32 of these filters.

# Convolutional Layer 3.
filter_size4 = 4          # Convolution filters are 2 x 2 pixels.
num_filters4 = 32         # There are 64 of these filters.

filter_size5 = 2          # Convolution filters are 2 x 2 pixels.
num_filters5 = 16         # There are 64 of these filters.


# Fully-connected layer.
fc_size = 2000             # Number of neurons in fully-connected layer.

# We know that images are 60 pixels in each dimension.
# img_size = 8 * 4

# Images are stored in one-dimensional arrays of this length.

# Number of colour channels for the images: 3 channel for RGB.
num_channels = 3

# Tuple with height and width of images used to reshape arrays.
img_shape = (7, 7, num_channels)

# Number of classes, one class for same or different image
num_classes = img_shape[0]*img_shape[1]
orig_patch_size = (2, 2, 3)
npatches = 1

"""Load Data and other functions"""
def change_label_dimensions(labels):
    label_temp = np.zeros((len(labels), 2))
    
    for idx in range(0, len(labels)):
        if labels[idx] == 1:
            label_temp[idx][1] = 1
        else:
            label_temp[idx][0] = 1

    
    return label_temp

def load_data(img_dir):
        list_of_orig_imgs = []
        list_of_labels = []
        list_same_diff = []
        list_img_keys = []
        for img in os.listdir(img_dir):
            
            img_path = os.path.join(img_dir, img)
            list_same_diff.append(int(os.listdir(img_path)[0]))
            list_img_keys.append(img)
            img_path = img_path + "/" + os.listdir(img_path)[0]
            for img_label in os.listdir(img_path):
                img_data = os.path.join(img_path, img_label)
                if img_label == "img":
#                     print(img_data + "/img.png")
                    list_of_orig_imgs.append(img_data + "/img.png")
                else:
                    list_of_labels.append([os.path.join(img_data, label) for label in os.listdir(img_data)])

        data_imgs = np.array(list_of_orig_imgs)
        data_labels = np.array(list_of_labels)
        data_same_diff = np.array(list_same_diff)
        data_img_keys = np.array(list_img_keys)

        return data_imgs, data_labels, data_same_diff, data_img_keys

    
def get_batch_images(data_orig, label, same_diff, img_keys, rshp, grey_scale):
        list_of_orig_imgs = []
        list_of_labels = []
        list_of_same_diff = []
        list_of_img_keys = []
        for img_orig, lbl, img_type, img_key in zip(data_orig, label, same_diff, img_keys):
            if (grey_scale):
                orig_img = cv2.imread(img_orig)
            
            orig_lbl = cv2.imread(lbl, cv2.IMREAD_GRAYSCALE)
            if orig_img is None or orig_lbl is None:
                    print ("Unable to read image{} or {}".format(img_orig, lbl))
                    continue
            
#             if (grey_scale):
#                 orig_lbl = rgb2grey(orig_lbl)

            flattened_orig_img = orig_img.flatten()
            flattened_lbl = orig_lbl.flatten()
            
#             if grey_scale:
#                 flattened_lbl = np.reshape(flattened_lbl, [10, 10])
#                 print(flattened_lbl)
#                 flattened_lbl = normalize(flattened_lbl)
#                 print(flattened_lbl)
            
            list_of_orig_imgs.append(np.asarray(flattened_orig_img, dtype=np.float32))
        
            list_of_labels.append(np.asarray(flattened_lbl, dtype=np.float32))
            list_of_same_diff.append(img_type)
            list_of_img_keys.append(img_key)

        data_labels = np.array(list_of_labels)
        data_imgs = np.array(list_of_orig_imgs)
        data_img_type = np.array(list_of_same_diff)
        data_img_keys = np.array(list_of_img_keys)
        
        """this function call locates top left location of each patch in the mask and return. Comment it if contents are required."""
#         print(data_labels.shape)
#         data_labels = get_patch_loc(data_labels)
        
        return data_imgs, data_labels, data_img_type, data_img_keys

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_orig = data
    data_orig_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_orig_shuffle), np.asarray(labels_shuffle)

"""Helpers functions for the Network"""
def new_weights(shape, layer_name):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape), name=layer_name+'_W')

def new_bias(length, layer_name):
    return tf.Variable(tf.constant(0.05, shape=[length]), name=layer_name+'_b')

def new_conv_layer(input,
                   num_input_channels,
                   filter_size,
                   num_filters,
                   name_scope,
                   layer_name='',
                   use_pooling=True):

    with tf.name_scope(name_scope):
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = new_weights(shape, layer_name)
        biases = new_bias(num_filters, layer_name)

        layer = tf.add(tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME'), biases, name=layer_name)

        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME', name=layer_name+'_max')
        layer = tf.nn.relu(layer, name=layer_name+'_activation')

    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    
    return layer_flat, num_features

def new_fc_layer(input,
                num_inputs,
                num_outputs,
                name_scope,
                layer_name='',
                use_relu=True):
    
    with tf.name_scope(name_scope):
        weights = new_weights([num_inputs, num_outputs], layer_name)
        biases = new_bias(num_outputs, layer_name)

        layer = tf.add(tf.matmul(input, weights),biases,name=layer_name)
    #     layer = tf.matmul(input, weights) + biases

        if use_relu:
            layer = tf.nn.relu(layer, layer_name+'_activation')
    
    return layer

def normalise(tensor):
    return tf.div(
   tf.subtract(
      tensor, 
      tf.reduce_min(tensor)
   ), 
   tf.subtract(
      tf.reduce_max(tensor), 
      tf.reduce_min(tensor)
   )
)

def normalized(arr):
    return (arr - np.min(arr))/(np.max(arr) - np.min(arr))

def optimize(num_epochs, save_model=True,save_name= "base_model",restore_model=False,restore_name=None):
    total_iterations = 0
    done_train_imgs = 0
    start_time = time.time()
    start_ = 0
    end_ = train_batch_size    
    plot_accuracy=[]
    plot_accuracy_epoch=[]
    plot_training_size=[]
    plot_training_size_epoch=[]
    saver = tf.train.Saver()
    sum_accuracy = 0.0
    n = 1
    
        #to save the model
    for i in range(0, num_epochs):   
        start_batch=0
        end_batch = train_batch_size
        
        print("Epoch:", i + 1)
        
        if restore_model==True:
            if restore_name==None:
                print("No model file specified")
                return
            else:
                saver.restore(session,restore_name)
        
        sum_accuracy = 0.0
        n = 1
        while end_batch < total_imgs:
            train_orig = train_orig_data[start_batch:end_batch]
            labels = train_labels_data[start_batch:end_batch]
#             print('Labels:', labels)
            img_type_lbl = img_type[start_:end_]
            img_key = img_keys[start_:end_]
            dims = (len(train_orig), num_classes, num_channels)
            train, labels, img_type_lbl, img_key = get_batch_images(train_orig, labels, img_type_lbl, img_key, dims, True)
            if not len(train) and not len(labels):
                print("All images have been processed.")
                break;

            x_orig_batch, y_true_batch = next_batch(len(train), train, labels)
            feed_dict_train = {x: x_orig_batch,
                       y_true: y_true_batch}
            
            session.run(optimizer, feed_dict=feed_dict_train)
    
            acc,co = session.run([accuracy, cost], feed_dict=feed_dict_train)
            sum_accuracy += acc
            n+=1
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}, Loss: {2:>.4f}"
            print(msg.format(end_batch + 1, acc, co))
            if i == num_epochs - 1:
                plot_accuracy.append(acc)
                plot_training_size.append(end_batch + 1)

            start_batch += train_batch_size
            end_batch += train_batch_size
    
        if save_model==True:
            if save_name==None:
                print("No model specified, model not being saved")
                return
            else:
                save_path = saver.save(session, save_name)
                restore_model = True
                print("Model saved in file: %s" % save_name)
        plot_accuracy_epoch.append(sum_accuracy/n)
        plot_training_size_epoch.append(i + 1)
    
    end_time = time.time()
    # Difference between start and end-times.
    time_dif = end_time - start_time
    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))  
    print(plot_accuracy)
    print(plot_training_size)
    print(plot_accuracy_epoch)
    print(plot_training_size_epoch)

def restore_see_layer(orig, model_name=None, var_name=None):
    with tf.Session('', tf.Graph()) as s:
        with s.graph.as_default():
            if ((model_name != None) and var_name != None):
                saver = tf.train.import_meta_graph(model_name+".meta")
                saver.restore(s, model_name)
#                 print(pred.shape)
                fd = {'x:0': orig}
#                 print(fd.shape)
                var_name=var_name+":0"
                
                result = 0
                result = s.run(var_name, feed_dict=fd)
    return result

def see_output_grey(iNp,depth_filter_to_see=0,cmap="gray",figsize=(4,4)):
    img_x = iNp[0,:,:]
    fig = plt.figure(figsize=figsize)
    plt.imshow(img_x, interpolation='none', aspect='auto')
#     plt.colorbar(img_x, orientation='horizontal')
    plt.show()


def see_output(iNp,depth_filter_to_see=0,cmap="gray",figsize=(4,4)):
    img_x = iNp[0,:,:,:]
    fig = plt.figure(figsize=figsize)
    if cmap == "gray":
        plt.imshow(img_x, cmap=plt.get_cmap('gray'))
    else:
        plt.imshow(img_x, interpolation='none', aspect='auto')
#     plt.colorbar(img_x, orientation='horizontal')
    plt.show()
    
def save_patch_images(img_x1, lbl_x1, index):
    if not os.path.exists('./SD/predicted_patches/' + str(index)):
        os.makedirs('./SD/predicted_patches/' + str(index))
        os.makedirs('./SD/predicted_patches/' + str(index) + "/" + str(lbl_x1))
        
    plt.imsave('./SD/predicted_patches/' + str(index) + "/" + str(lbl_x1) + '/img.png', np.squeeze(img_x1))
    

def predict_nd_save(train, labels, img_type_lbl, img_key, start_idx):

    for index in range(0, len(train)):
        img_x = train[index:index+1, :]
        lbl_x = labels[index:index+1, :]
        img_type_x = img_type_lbl[index]
        img_key_x = img_key[index]
        prediction = restore_see_layer(ix=img_x,model_name=model2_50000,var_name='Softmax')
        prediction = np.reshape(prediction, (1, 4, 2, 1))   
        save_patch_images(prediction, img_type_x, img_key_x)



x = tf.placeholder(tf.float32, shape=[None, img_shape[0]*img_shape[1]*num_channels], name='x')
x_image = tf.reshape(x, [-1, img_shape[0], img_shape[1], num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true_cls')

layer0_conv0, weights_conv0 = new_conv_layer(input=x_image,
                                        num_input_channels=num_channels,
                                        filter_size=filter_size0,
                                        num_filters=num_filters0,
                                        name_scope='conv',
                                        layer_name='conv1',
                                        use_pooling=True)


layer1_conv1, weights_conv1 = new_conv_layer(input=layer0_conv0,
                                        num_input_channels=num_filters0,
                                        filter_size=filter_size1,
                                        num_filters=num_filters1,
                                        name_scope='conv',
                                        layer_name='conv2',
                                        use_pooling=True)



layer2_conv2, weights_conv2 = new_conv_layer(input=layer1_conv1,
                                        num_input_channels=num_filters1,
                                        filter_size=filter_size2,
                                        num_filters=num_filters2,
                                        name_scope='conv',
                                        layer_name='conv3',
                                        use_pooling=True)



layer3_conv3, weights_conv3 = new_conv_layer(input=layer2_conv2,
                                        num_input_channels=num_filters2,
                                        filter_size=filter_size3,
                                        num_filters=num_filters3,
                                        name_scope='conv',
                                        layer_name='conv4',
                                        use_pooling=True)



layer4_conv4, weights_conv4 = new_conv_layer(input=layer3_conv3,
                                        num_input_channels=num_filters3,
                                        filter_size=filter_size4,
                                        num_filters=num_filters4,
                                        name_scope='conv',
                                        layer_name='conv5',
                                        use_pooling=True)



layer5_conv5, weights_conv5 = new_conv_layer(input=layer4_conv4,
                                        num_input_channels=num_filters4,
                                        filter_size=filter_size5,
                                        num_filters=num_filters5,
                                        name_scope='conv',
                                        layer_name='conv6',
                                        use_pooling=True)



layer_flat, num_features = flatten_layer(layer5_conv5)


layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         name_scope='fc',
                         layer_name='fc1',
                         use_relu=True)



layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=fc_size,
                         name_scope='fc',
                         layer_name='fc2',
                         use_relu=True)



layer_fc3 = new_fc_layer(input=layer_fc2,
                         num_inputs=fc_size,
                         num_outputs=fc_size,
                         name_scope='fc',
                         layer_name='fc3',
                         use_relu=True)



layer_fc4 = new_fc_layer(input=layer_fc3,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         name_scope='fc',
                         layer_name='fc4',
                         use_relu=True)


y_pred = layer_fc4

cost = tf.reduce_mean(tf.square(y_true - y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# ## some more performance measures
correct_prediction = tf.equal(y_pred, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


train_data, train_labels, img_type, img_keys = load_data(img_dir)
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
train_labels_data = train_labels[:][:, 1]#1=mask_patch_2,2=mask_patch_1
#print(train_labels_data)
#sys.exit(0)
train_orig_data = train_data[:]
img_type = img_type[:]
img_keys = img_keys[:]
total_imgs = len(img_type)
train_batch_size = 64 


"""Main"""
if __name__ == "__main__":

    save_model = True
    save_name = model50_right_mask
    restore_model=False
    restore_name=model50_right_mask

    optimize(5,
    save_model=True,save_name=model50_right_mask,restore_model=restore_model,restore_name=model50_right_mask)
