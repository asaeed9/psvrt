from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import cv2, os, math, time, sys
from datetime import timedelta
from sklearn.utils import shuffle
from numpy.lib.stride_tricks import as_strided
from functools import reduce
from itertools import product


###Configuration
"""
Data Configurations/Paths
"""
img_dir="../../original_images/SD"
model7m_sr = 'SD/model20m_sr.ckpt'

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
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            predicted_msk = cv2.imread(img_pred, cv2.IMREAD_GRAYSCALE)
        else:
            orig_img = cv2.imread(img_pred[0])

        # print(lbl, type(lbl))
        orig_lbl = cv2.imread(lbl, cv2.IMREAD_GRAYSCALE)
        if orig_img is None or orig_lbl is None:
            print("Unable to read image{} or {}".format(img_pred[0], lbl))
            continue

        # if (grey_scale):
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

        if img_type == 1:  # 0 is same and 1 is different
            list_of_same_diff.append([1, 0])
        else:
            list_of_same_diff.append([0, 1])

        # list_of_same_diff.append(img_type)

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
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


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


def extract_patch(locs, orig_batch):
    orig_batch = np.reshape(orig_batch, [-1,10,10,3])
    return np.array([orig_batch[idx:idx+1, loc[0]:loc[0]+orig_patch_size[0], loc[1]:loc[1]+orig_patch_size[1], :] for idx, loc in enumerate(locs)])


def get_accuracy(nimgs, img_data_loc, model):
    train_data, train_labels, img_type, img_keys = load_data(img_data_loc)

    if nimgs == -1:
        nimgs = len(train_data[0, :])

    train_orig_data = train_data[0, :nimgs]
    total_imgs = len(train_orig_data)

    print("Total Images:", total_imgs)

    cor_prd_imgs = 0

    #     if category == "both":
    #         train_mask_labels = train_labels[:nimgs, 5]
    train_lft_labels = train_labels[:nimgs, 1]  # 2=mask_patch_2, 1=mask_patch_1
    # train_rght_labels = train_labels[:nimgs, 2]

    #     print(train_lft_labels)
    #     print(train_rght_labels)
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
        # rght_labels = train_rght_labels[start_:end_]

        img_type_lbl = img_type[start_:end_]
        img_key = img_keys[start_:end_]
        dims = (batch_s, num_classes, num_channels)
        train, labels, img_type_lbl_lft, img_key_lft = get_batch_images(train, train, lft_labels, img_type_lbl,
                                                                                img_key, dims, True)

        x_orig_batch, true_batch = next_batch(len(train[0]), train, labels)
        mod_preds = restore_see_layer(orig=x_orig_batch, model_name=model, var_name='fc_3/fc4')

        cor_prd_imgs += np.count_nonzero(mod_preds[:, 0] == true_batch[:, 0])

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
    # np.random.seed(101)
    #
    # ar1 = np.random.randint(2, size=(10, 1))
    # print(ar1)
    # ar11 = ar1^1
    # # print(ar11)
    # ar1 = np.hstack((ar1, ar11))
    # ar2 = np.random.randint(2, size=(10, 1))
    # print(ar2)
    # ar21 = ar2^1
    # ar2 = np.hstack((ar2, ar21))
    # tp = np.count_nonzero(ar1[:, 0] == ar2[:, 0])
    # total = len(ar1)
    #
    # print("Accuracy: ", tp/total)
    np.set_printoptions(suppress=True)
    ary = np.array([[0.5, 0.5],
                    [0.00011721, 0.9998828 ],
 [0.5,        0.5       ],
 [0.54142386, 0.45857617],
 [0.5,        0.5       ]])

    pred_lbls = np.zeros(ary.shape)
    pred_lbls[np.arange(ary.shape[0]), np.argmax(ary, axis=1)] = 1
    print(pred_lbls)

    # get_accuracy(-1, "../../original_images/SD", model7m_sr)