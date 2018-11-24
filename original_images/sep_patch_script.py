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


"""
Data Configurations/Paths
"""
img_dir="./SD/"
img_lbls=""
base_model = 'SD/sd_01_.ckpt'
model_20000 = 'SD/sd_20000.ckpt'
model_30000 = 'SD/sd_30000.ckpt'
model_40000 = 'SD/sd_40000.ckpt'
model_50000 = 'SD/sd_50000.ckpt'
model2_50000 = 'SD/sd2_50000.ckpt'

connected_model = 'SD/sd_conected.ckpt'

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
fc_size = 2000             # Number of neurons in fully-connected layer.

# We know that images are 60 pixels in each dimension.
# img_size = 8 * 4

# Images are stored in one-dimensional arrays of this length.
img_size_flat = 10 * 10

# Number of colour channels for the images: 3 channel for RGB.
num_channels = 1

# Tuple with height and width of images used to reshape arrays.
img_shape = (10, 10, num_channels)

# Number of classes, one class for same or different image
num_classes = 4*2
patch_size = (2, 2)
npatches = 2

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

def rgb2grey(rgb):
    return(np.dot(rgb[...,:3], [0.299, 0.587, 0.114]))

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

def get_batch_labels(label, same_diff, img_keys, rshp, grey_scale):
    list_of_labels = []
    list_of_same_diff = []
    list_of_img_keys = []
    for lbl, img_type, img_key in zip(label, same_diff, img_keys):
        orig_lbl = cv2.imread(lbl)
        if orig_lbl is None:
            print("Unable to read label{}".format(lbl))
            continue

        if (grey_scale):
            orig_lbl = rgb2grey(orig_lbl)

        flattened_lbl = orig_lbl.flatten()

        if grey_scale:
            flattened_lbl = np.reshape(flattened_lbl, [10, 10])

        list_of_labels.append(np.asarray(flattened_lbl, dtype=np.float32))
        list_of_same_diff.append(img_type)
        list_of_img_keys.append(img_key)

    data_labels = np.array(list_of_labels)
    data_img_type = np.array(list_of_same_diff)
    data_img_keys = np.array(list_of_img_keys)

    return data_labels, data_img_type, data_img_keys

def determine_quad(left_top_locations, mid_point):
    box_1 = left_top_locations[0]
    box_2 = left_top_locations[1]
    box_quad = []
    quad_dict = {}

    for box in left_top_locations:
        if box[1] <= mid_point[0] and box[0] <= mid_point[1]:  # both x and y are less than equal to the midpoint
            #         print("Quadrant II")
            box_quad.append(2)
        elif box[1] <= mid_point[0] and box[0] > mid_point[
            1]:  # x is less than equal to x of midpoint but y is greater than y of midpoint
            #         print("Quadrant III")
            box_quad.append(3)
        elif box[1] > mid_point[0] and box[0] <= mid_point[
            1]:  # x is greater than x of midpoint but y is less than equal to y of midpoint
            #         print("Quadrant I")
            box_quad.append(1)
        elif box[1] > mid_point[0] and box[0] > mid_point[
            1]:  # x is greater than x of midpoint and y is greater than  y of midpoint
            #         print("Quadrant IV")
            box_quad.append(4)

            #     print('quadrant:', box_quad)

    if box_quad[0] == box_quad[1]:
        if abs(box_1[1] - box_2[1]) > abs(box_1[0] - box_2[0]):  # top/bottom case
            if box_1[1] < box_2[1]:
                quad_dict['left_box'] = box_1
                quad_dict['right_box'] = box_2
            else:
                quad_dict['left_box'] = box_2
                quad_dict['right_box'] = box_1

        else:  # left/right case
            if box_1[0] < box_2[0]:
                quad_dict['top_box'] = box_1
                quad_dict['bottom_box'] = box_2
            else:
                quad_dict['top_box'] = box_2
                quad_dict['bottom_box'] = box_1

    if (box_quad[0] == 2 and box_quad[1] == 1) or (box_quad[0] == 2 and box_quad[1] == 4) or (
            box_quad[0] == 3 and box_quad[1] == 1) or (box_quad[0] == 3 and box_quad[1] == 4):
        #     0 == Left and 1 == Right
        #         print('Left box:', box_1)
        #         print('Right box:', box_2)
        quad_dict['left_box'] = box_1
        quad_dict['right_box'] = box_2

    if (box_quad[0] == 1 and box_quad[1] == 2) or (box_quad[0] == 4 and box_quad[1] == 2) or (
            box_quad[0] == 1 and box_quad[1] == 3) or (box_quad[0] == 4 and box_quad[1] == 3):
        #    1 == Left and 0 == Right
        #         print('Left box:', box_2)
        #         print('Right box:', box_1)
        quad_dict['left_box'] = box_2
        quad_dict['right_box'] = box_1

    if (box_quad[0] == 1 and box_quad[1] == 4) or (box_quad[0] == 2 and box_quad[1] == 3):
        #    0 == Top and 1 == Bottom
        #         print('Top box:', box_1)
        #         print('Bottom box:', box_2)
        quad_dict['top_box'] = box_1
        quad_dict['bottom_box'] = box_2

    if (box_quad[0] == 4 and box_quad[1] == 1) or (box_quad[0] == 3 and box_quad[1] == 2):
        #    1 == Top and 0 == Bottom
        #         print('Top box:', box_2)
        #         print('Bottom box:', box_1)
        quad_dict['top_box'] = box_2
        quad_dict['bottom_box'] = box_1

    return quad_dict

def get_patch_loc(img):
    img = np.squeeze(img)
    img[np.where(img == 255)] = 1.
    r = img_shape[0] - patch_size[0] + 1
    c = img_shape[1] - patch_size[0] + 1

    iter_o = np.array(list(product(range(0,patch_size[0]), range(0,patch_size[1]))))
    left_corners = reduce(lambda x,y: np.multiply(x,y), map(lambda x, y: img[x:x+r, y:y+c], iter_o[:, 0], iter_o[:, 1]))
    left_top_locations = np.transpose(np.where(left_corners == 1))
    left_top_locations = np.array(left_top_locations, dtype=np.int32)
    return left_top_locations

def sep_boxes(labels, img_type_lbl, img_key, start_idx):
    patched_images = []
    mid_point = (4, 4)
    for index in range(0, len(labels)):
        lbl_x = labels[index:index + 1, :]
        img_type_x = img_type_lbl[index]
        img_key_x = img_key[index]
        top_left_loc = get_patch_loc(lbl_x)
        quad_dict = determine_quad(top_left_loc, mid_point)

        mask_1 = np.zeros(100)
        mask_2 = np.zeros(100)
        mask_1 = np.reshape(mask_1, [10, 10])
        mask_2 = np.reshape(mask_2, [10, 10])

        if 'left_box' in quad_dict:
            coords = quad_dict['left_box']
            mask_1[coords[0]: coords[0] + 2, coords[1]: coords[1] + 2] = 255.
            coords = quad_dict['right_box']
            mask_2[coords[0]:coords[0] + 2, coords[1]: coords[1] + 2] = 255.
        elif 'top_box' in quad_dict:
            coords = quad_dict['top_box']
            mask_1[coords[0]: coords[0] + 2, coords[1]: coords[1] + 2] = 255.
            coords = quad_dict['bottom_box']
            mask_2[coords[0]:coords[0] + 2, coords[1]: coords[1] + 2] = 255.

        plt.imsave(img_dir + str(img_key_x) + "/" + str(img_type_x) + '/labels/mask_patch_1.png',
                   np.squeeze(mask_1))
        plt.imsave(img_dir + str(img_key_x) + "/" + str(img_type_x) + '/labels/mask_patch_2.png',
                   np.squeeze(mask_2))

    return patched_images


if __name__ == "__main__":

    train_data, train_labels, img_type, img_keys = load_data(img_dir)
    # train_orgs = train_data[0, :]
    # train_prds = train_data[1, :]
    train_mask_labels = train_labels[:, 1]
    batch_s = 64
    total_iterations = 0
    start_ = 0
    end_ = batch_s
    np.set_printoptions(suppress=True)
    while True:

        #     train_org = train_orgs[start_:end_]
        #     train_prd = train_prds[start_:end_]
        mask_lbls = train_mask_labels[start_:end_]
        img_type_lbl = img_type[start_:end_]
        img_key = img_keys[start_:end_]
        dims = (batch_s, num_classes, num_channels)
        labels, img_type_lbl, img_key = get_batch_labels(mask_lbls, img_type_lbl, img_key, dims, True)
        patch_images = sep_boxes(labels, img_type_lbl, img_key, start_)

        # do my stuff
        if len(train_mask_labels) < start_ + batch_s:
            print("{} Images have been processed.".format(total_iterations))
            break

        total_iterations += batch_s
        start_ = end_
        end_ = end_ + batch_s