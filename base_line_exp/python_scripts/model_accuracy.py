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
img_dir="../../original_images/SD_Train"
model50_SD = 'SD/50kSD_Model.ckpt'
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
img_size_flat = 128 * 128

# Number of colour channels for the images: 3 channel for RGB.
num_channels = 3

# Tuple with height and width of images used to reshape arrays.
img_shape = (128, 128, num_channels)

# Number of classes, one class for same or different image
num_classes = 128*128
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


def patch_next_batch(num, data, lft_lbls, rght_lbls, img_tlbl, img_key):
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
    lft_labels = [lft_lbls[ i] for i in idx]
    rght_labels = [rght_lbls[ i] for i in idx]
    img_tlbl = [img_tlbl[ i] for i in idx]
    img_key = [img_key[ i] for i in idx]

    return np.asarray(data_orig_shuffle), np.asarray(data_pred_shuffle), np.asarray(lft_labels), np.asarray(rght_labels), np.asarray(img_tlbl), np.asarray(img_key)

def save_patch_images(img_x1, lbl_x1, index):
    if not os.path.exists('./SD/predicted_patches/' + str(index)):
        os.makedirs('./SD/predicted_patches/' + str(index))
        os.makedirs('./SD/predicted_patches/' + str(index) + "/" + str(lbl_x1))

    plt.imsave('./SD/predicted_patches/' + str(index) + "/" + str(lbl_x1) + '/img.png', np.squeeze(img_x1))

def restore_see_layer(orig, model_name=None, var_name=None):
    with tf.Session('', tf.Graph()) as s:
        with s.graph.as_default():
            if ((model_name != None) and var_name != None):
                saver = tf.train.import_meta_graph(model_name + ".meta")
                saver.restore(s, model_name)
                fd = {'x:0': orig}
                #                 print(fd.shape)
                var_name = var_name + ":0"

                result = 0
                result = s.run(var_name, feed_dict=fd)
    return result


def get_coords(imgs, top):
    # Input data
    input = tf.placeholder(tf.float32, [None])
    orig_input = tf.placeholder(tf.float32, [None])
    # Submatrix dimension
    dims = tf.placeholder(tf.int32, [None])
    # Number of top submatrices to find
    k = tf.placeholder(tf.int32, [])

    my_inp = tf.reshape(input, [-1, img_shape[0], img_shape[1]])
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
    subm = tf.gather_nd(my_inp[0, :, :], tf.stack([subm_ii, subm_jj], axis=-1))
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
            res = sess1.run(result, feed_dict={input: img, orig_input: img, dims: lbl_patch_size, k: top})
            lst_cds.append(np.squeeze(res))

            #         print(lst_cds)
    return np.array(lst_cds)


def extract_patch(locs, orig_batch):
    orig_batch = np.reshape(orig_batch, [-1,img_shape[0],img_shape[1],3])
#     print(orig_batch)
#     print(locs)
#     return np.array([loc for loc in locs])
    return np.array([orig_batch[idx:idx+1, loc[0]:loc[0]+orig_patch_size[0], loc[1]:loc[1]+orig_patch_size[1], :] for idx, loc in enumerate(locs)])


def get_accuracy(nimgs, img_data_loc, lft_model, rght_model):
    train_data, train_labels, img_type, img_keys = load_data(img_data_loc)

    if nimgs == -1:
        nimgs = len(train_data[0, :])

    train_orig_data = train_data[0, :nimgs]
    total_imgs = len(train_orig_data)

    print("Total Images:", total_imgs)

    cor_prd_imgs = 0

    #     if category == "both":
    #         train_mask_labels = train_labels[:nimgs, 5]
    train_lft_labels = train_labels[:nimgs, 2]  # 2=mask_patch_2, 1=mask_patch_1
    train_rght_labels = train_labels[:nimgs, 1]

    print(train_lft_labels)
    print(train_rght_labels)
    #     print(train_mask_labels)
    #     train_patch_labels = train_labels[:nimgs, 0]
    batch_s = 256
    total_iterations = 0
    start_ = 0
    end_ = batch_s
    np.set_printoptions(suppress=True)

    while True:
        train = train_orig_data[start_:end_]
        lft_labels = train_lft_labels[start_:end_]
        rght_labels = train_rght_labels[start_:end_]

        img_type_lbl = img_type[start_:end_]
        img_key = img_keys[start_:end_]
        dims = (batch_s, num_classes, num_channels)
        train_lft, lft_labels, img_type_lbl_lft, img_key_lft = get_batch_images(train, train, lft_labels, img_type_lbl,
                                                                                img_key, dims, True)
        train, rght_labels, img_type_lbl, img_key = get_batch_images(train, train, rght_labels, img_type_lbl, img_key,
                                                                     dims, True)

        x_orig_batch, x_pred_batch, lft_true_batch, rght_true_batch, img_type_lbl, img_key = patch_next_batch(
            len(train[0]), train, lft_labels, rght_labels, img_type_lbl, img_key)
        lft_preds = restore_see_layer(orig=x_orig_batch, model_name=lft_model, var_name='fc_3/fc4')
        rght_preds = restore_see_layer(orig=x_orig_batch, model_name=rght_model, var_name='fc_3/fc4')
        rght_preds_w = restore_see_layer(orig=x_orig_batch,
        model_name=rght_model, var_name='fc_2/fc3_W')
        print(rght_preds_w)
        lft_lbl_cds = get_coords(lft_true_batch, 1)
        rght_lbl_cds = get_coords(rght_true_batch, 1)
        lft_pred_cds = get_coords(lft_preds, 1)
        rght_pred_cds = get_coords(rght_preds, 1)

        print(lft_lbl_cds)
        print("******left predicted")
        print(lft_pred_cds)
        print("******")
        print(rght_lbl_cds)
        print("******right predicted")
        print(rght_pred_cds)

        left_patch = extract_patch(lft_pred_cds, x_orig_batch)
        right_patch = extract_patch(rght_pred_cds, x_orig_batch)

        cor_prd_imgs += np.sum(
            [True for llc, lpc, rlc, rpc in zip(lft_lbl_cds, lft_pred_cds, rght_lbl_cds, rght_pred_cds) if
             np.array_equal(llc, lpc) and np.array_equal(rlc, rpc)])

        #         for img0,img_key_x in zip(x_orig_batch, img_key):
        #             print('key:', img_key_x)
        #             see_output(np.expand_dims(np.reshape(img0, [10,10,3]), axis=0))
        #             see_output_grey(np.expand_dims(np.reshape(np.rint(img1), [10,10]), axis=0))
        #             see_output_grey(np.expand_dims(np.reshape(np.rint(img2), [10,10]), axis=0))

        for lft, rght, img_key_x, img_type_x in zip(left_patch, right_patch, img_key, img_type_lbl):
            prediction = np.reshape(np.stack([lft, rght], axis=0), [1, 4, 2, 3])
            save_patch_images(prediction, img_type_x, img_key_x)
        # see_output(np.reshape(np.stack([lft,rght], axis=0), [1,4,2,3]))
        #             see_output(lft)
        #             see_output(rght)


        # do my stuff
        if total_imgs <= start_ + len(train[0]):
            total_iterations += len(train[0])
            print("{} Image(s) have been processed.".format(total_iterations))
            break

        total_iterations += batch_s
        start_ = end_
        end_ = end_ + batch_s

    return (cor_prd_imgs / total_imgs)


if __name__ == "__main__":
    np.random.seed(100)
    print(get_accuracy(256, "../../original_images/SD", model50_left_mask,
    model50_right_mask))
