from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2, os, sys, math, time
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
model50_local_top_1 = 'SD/50k_local_top_Model.ckpt'
model50_left_mask = 'SD/50k_left_mask.ckpt'
model50_right_mask = 'SD/50k_right_mask.ckpt'

img_type = "original"
# img_type = "patch"

##
# Convolutional Layer 1.
filter_size0 = 16          # Convolution filters are 4 x 4 pixels.
num_filters0 = 16         # There are 16 of these filters.

filter_size1 = 8          # Convolution filters are 4 x 4 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 8          # Convolution filters are 2 x 2 pixels.
num_filters2 = 16         # There are 32 of these filters.

filter_size3 = 8          # Convolution filters are 2 x 2 pixels.
num_filters3 = 4         # There are 32 of these filters.

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
img_size_flat = 10 * 10

# Number of colour channels for the images: 3 channel for RGB.
num_channels = 3

# Tuple with height and width of images used to reshape arrays.
img_shape = (10, 10, num_channels)

# Number of classes, one class for same or different image
num_classes = 10*10
orig_patch_size = (2, 2, 3)
lbl_patch_size = (2, 2)
npatches = 1

def load_data(img_dir):
        list_of_orig_imgs = []
        list_of_pred_imgs = []
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
                    list_of_pred_imgs.append(img_data + "/predicted_mask.png")
                else:
                    list_of_labels.append([os.path.join(img_data, label) for label in os.listdir(img_data)])

        data_imgs = np.array([list_of_orig_imgs, list_of_pred_imgs])
        data_labels = np.array(list_of_labels)
        data_same_diff = np.array(list_same_diff)
        data_img_keys = np.array(list_img_keys)

        return data_imgs, data_labels, data_same_diff, data_img_keys

    
def get_batch_images(data_orig, data_pred, label, same_diff, img_keys, rshp, grey_scale):
        list_of_orig_imgs = []
        list_of_pred_imgs = []
        list_of_labels = []
        list_of_same_diff = []
        list_of_img_keys = []
        for img_orig, img_pred, lbl, img_type, img_key in zip(data_orig, data_pred, label, same_diff, img_keys):
            if (grey_scale):
                orig_img = cv2.imread(img_orig)
                predicted_msk = cv2.imread(img_pred, cv2.IMREAD_GRAYSCALE)
            else:
                orig_img = cv2.imread(img[0])
            
#             print(lbl, type(lbl))
            orig_lbl = cv2.imread(lbl, cv2.IMREAD_GRAYSCALE)
            if orig_img is None or orig_lbl is None:
                    print ("Unable to read image{} or {}".format(img[0], lbl))
                    continue
            
#             if (grey_scale):
#                 orig_lbl = rgb2grey(orig_lbl)

            flattened_orig_img = orig_img.flatten()
            flattened_pred_img = predicted_msk.flatten()
            flattened_lbl = orig_lbl.flatten()
            
#             if grey_scale:
#                 flattened_lbl = np.reshape(flattened_lbl, [10, 10])
#                 print(flattened_lbl)
#                 flattened_lbl = normalize(flattened_lbl)
#                 print(flattened_lbl)
            
            list_of_orig_imgs.append(np.asarray(flattened_orig_img, dtype=np.float32))
            list_of_pred_imgs.append(np.asarray(flattened_pred_img, dtype=np.float32))
        
            list_of_labels.append(np.asarray(flattened_lbl, dtype=np.float32))
            list_of_same_diff.append(img_type)
            list_of_img_keys.append(img_key)

        data_labels = np.array(list_of_labels)
        data_imgs = np.array([list_of_orig_imgs, list_of_pred_imgs])
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
    idx = np.arange(0 , len(data[0]))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_orig = data[0, :]
    data_pred = data[1, :]
    data_orig_shuffle = [data[0, i] for i in idx]
    data_pred_shuffle = [data[1, i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_orig_shuffle), np.asarray(data_pred_shuffle), np.asarray(labels_shuffle)


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

def get_predictions(train, labels, img_type_lbl, img_key, start_idx, model):
    list_preds = []
    for index in range(0, len(train)):
        img_x = train[index:index+1]
        lbl_x = labels[index:index+1]
        img_type_x = img_type_lbl[index:index+1]
        img_key_x = img_key[index:index+1]
        prediction = restore_see_layer(orig=np.expand_dims(img_x[0], axis=0),model_name=model,var_name='fc_3/fc4')
        list_preds.append(np.squeeze(prediction))
    
    return np.array(list_preds)

def get_coords(imgs):

    # Input data
    input = tf.placeholder(tf.float32, [None])
    orig_input = tf.placeholder(tf.float32, [None])
    # Submatrix dimension
    dims = tf.placeholder(tf.int32, [None])
    # Number of top submatrices to find
    k = tf.placeholder(tf.int32, [])

    my_inp = tf.reshape(input, [-1, 10,10])
    input_shape = tf.shape(my_inp)
    rows, cols = input_shape[1], input_shape[2]
    d_rows, d_cols = dims[0], dims[1]
    subm_rows, subm_cols = rows - d_rows + 1, cols - d_cols + 1

    # # Index grids
    ii, jj = tf.meshgrid(tf.range(subm_rows), tf.range(subm_cols), indexing='ij')
    d_ii, d_jj = tf.meshgrid(tf.range(d_rows), tf.range(d_cols), indexing='ij')
    # # # Add indices
    subm_ii = ii[:, :, tf.newaxis, tf.newaxis] + d_ii
    subm_jj = jj[:, :, tf.newaxis, tf.newaxis] + d_jj


    subm_st = tf.stack([subm_ii, subm_jj], axis=-1)
    # # Make submatrices tensor
    subm = tf.gather_nd(my_inp[0,:,:], tf.stack([subm_ii, subm_jj], axis=-1))
    # # Add submatrices
    subm_sum = tf.reduce_sum(subm, axis=(2, 3))
    # # Use TopK to find top submatrices
    _, top_idx = tf.nn.top_k(tf.reshape(subm_sum, [-1]), tf.minimum(k, tf.size(subm_sum)))
    # # # Get row and column
    top_row = top_idx // subm_cols
    top_col = top_idx % subm_cols
    result = tf.stack([top_row, top_col], axis=-1)
    
    with tf.Session() as sess1:
        lst_cds = []
        for img in imgs: 
            res = sess1.run(result, feed_dict={input: img, orig_input:img,  dims: lbl_patch_size, k: 1})
            lst_cds.append(res)

    return np.array(lst_cds)

def get_accuracy(nimgs, category, img_data_loc, model):
    train_data, train_labels, img_type, img_keys = load_data(img_data_loc)

    if nimgs == -1:
        nimgs = len(train_data[0, :])
        
    train_orig_data = train_data[0, :nimgs]
    total_imgs = len(train_orig_data)

    cor_prd_imgs = 0
    
    if category == "both":
        train_mask_labels = train_labels[:nimgs, 0]
#        print(train_mask_labels)
    elif category == "left":
        train_mask_labels = train_labels[:nimgs, 2]
        #1=mask_patch_2,2=mask_patch_1, 3=merged_patch 
    elif category == "right":
        train_mask_labels = train_labels[:nimgs, 1]

    
    batch_s = 64
    total_iterations = 0
    start_ = 0
    end_ = batch_s
    np.set_printoptions(suppress=True)

    while True:
        train = train_orig_data[start_:end_]
        labels = train_mask_labels[start_:end_]
        img_type_lbl = img_type[start_:end_]
        img_key = img_keys[start_:end_]
        dims = (batch_s, num_classes, num_channels)
        train, labels, img_type_lbl, img_key = get_batch_images(train, train, labels, img_type_lbl, img_key, dims, True)

        x_orig_batch, x_pred_batch, y_true_batch = next_batch(len(train[0]), train, labels)        
        preds = restore_see_layer(orig=x_orig_batch,model_name=model,var_name='fc_3/fc4')
        print(preds.shape)
        print(np.reshape(y_true_batch[2], [10,10]))
        print(np.reshape(np.rint(preds[2]), [10,10]))
        lbl_cds = get_coords(y_true_batch)
        pred_cds = get_coords(preds)
        
        print(lbl_cds)
        print(pred_cds)

        batch_output = np.array(list(map(lambda lbl, pred: np.array_equal(lbl, pred), lbl_cds, pred_cds)))
        cor_prd_imgs += np.sum(batch_output)
#        see_output_grey(np.expand_dims(np.reshape(np.rint(preds[0]), [10,10]), axis=0))
        
        #do my stuff
        if total_imgs <= start_ + batch_s:
            print("{} Images have been processed.".format(total_iterations))
            break

        total_iterations +=batch_s
        start_ = end_
        end_ = end_ + batch_s

    return(cor_prd_imgs/total_imgs)


if __name__ == "__main__":

    print(get_accuracy(int(sys.argv[2]), sys.argv[1],
    "../../original_images/SD", model50_right_mask))
