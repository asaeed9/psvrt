import tensorflow as tf
import numpy as np
import sys
import os
import subprocess
import webbrowser
import shutil
from tensorflow.python.client import device_lib
import ipdb
from functools import reduce
from itertools import product


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False


def concatenate_path_list(lst):
    cct = lst[0]
    for l in lst[1:]:
        cct += '/' + l
    return cct


def find_files(files, dirs=[], contains=[]):
    for d in dirs:
        onlyfiles = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        for i, part in enumerate(contains):
            files += [os.path.join(d, f) for f in onlyfiles if part in f]
        onlydirs = [os.path.join(d, dd) for dd in os.listdir(d) if os.path.isdir(os.path.join(d, dd))]
        if onlydirs:
            files += find_files([], onlydirs, contains)

    return files


def pairwise_add(u, v=None, is_batch=False):
    """
    performs a pairwise summation between vectors (possibly the same)
    Parameters:
    ----------
    u: Tensor (n, ) | (n, 1)
    v: Tensor (n, ) | (n, 1) [optional]
    is_batch: bool
        a flag for whether the vectors come in a batch
        ie.: whether the vectors has a shape of (b,n) or (b,n,1)
    Returns: Tensor (n, n)
    Raises: ValueError
    """
    u_shape = u.get_shape().as_list()

    if len(u_shape) > 2 and not is_batch:
        raise ValueError("Expected at most 2D tensors, but got %dD" % len(u_shape))
    if len(u_shape) > 3 and is_batch:
        raise ValueError("Expected at most 2D tensor batches, but got %dD" % len(u_shape))

    if v is None:
        v = u
    else:
        v_shape = v.get_shape().as_list()
        if u_shape != v_shape:
            raise ValueError("Shapes %s and %s do not match" % (u_shape, v_shape))

    n = u_shape[0] if not is_batch else u_shape[1]

    column_u = tf.reshape(u, (-1, 1) if not is_batch else (-1, n, 1))
    # U = tf.concat(1 if not is_batch else 2, [column_u] * n)
    U = tf.concat([column_u] * n, 1 if not is_batch else 2)
    if v is u:
        return U + tf.transpose(U, None if not is_batch else [0, 2, 1])
    else:
        row_v = tf.reshape(v, (1, -1) if not is_batch else (-1, 1, n))
        V = tf.concat(0 if not is_batch else 1, [row_v] * n)

        return U + V


def decaying_softmax(shape, axis):
    rank = len(shape)
    max_val = shape[axis]

    weights_vector = np.arange(1, max_val + 1, dtype=np.float32)
    weights_vector = weights_vector[::-1]
    weights_vector = np.exp(weights_vector) / np.sum(np.exp(weights_vector))

    container = np.zeros(shape, dtype=np.float32)
    broadcastable_shape = [1] * rank
    broadcastable_shape[axis] = max_val

    return container + np.reshape(weights_vector, broadcastable_shape)


def unpack_into_tensorarray(value, axis, size=None):
    """
    unpacks a given tensor along a given axis into a TensorArray
    Parameters:
    ----------
    value: Tensor
        the tensor to be unpacked
    axis: int
        the axis to unpack the tensor along
    size: int
        the size of the array to be used if shape inference resulted in None
    Returns: TensorArray
        the unpacked TensorArray
    """

    shape = value.get_shape().as_list()
    rank = len(shape)
    dtype = value.dtype
    array_size = shape[axis] if not shape[axis] is None else size

    if array_size is None:
        raise ValueError("Can't create TensorArray with size None")
    array = tf.TensorArray(dtype=dtype, size=array_size)
    dim_permutation = [axis] + range(0, axis) + range(axis + 1, rank)
    unpack_axis_major_value = tf.transpose(value, dim_permutation)
    full_array = array.unstack(unpack_axis_major_value)

    return full_array


def pack_into_tensor(array, axis):
    """
    packs a given TensorArray into a tensor along a given axis
    Parameters:
    ----------
    array: TensorArray
        the tensor array to pack
    axis: int
        the axis to pack the array along
    Returns: Tensor
        the packed tensor
    """

    packed_tensor = array.stack()

    shape = packed_tensor.get_shape()

    rank = len(shape)

    dim_permutation = [axis] + range(0, axis) + range(axis + 1, rank)

    correct_shape_tensor = tf.transpose(packed_tensor, dim_permutation)

    return correct_shape_tensor


def pack_into_tensor_sized(array, axis, size):
    tensor_list = []

    for ss in range(size):
        tensor = array.read(ss)
        tensor_list.append(tensor)

    stacked_tensor = tf.stack(tensor_list)
    shape = stacked_tensor.get_shape()
    rank = len(shape)

    dim_permutation = [axis] + range(0, axis) + range(axis + 1, rank)
    correct_shape_tensor = tf.transpose(stacked_tensor, dim_permutation)

    return correct_shape_tensor


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def prepare_summary(value, sum_name, sum_type):
    shape = value.get_shape().as_list()
    rank = len(shape)
    if sum_type is "Image":
        if rank is 5:
            if shape[3] == 1:
                value = tf.squeeze(value, axis=3)
            else:
                value = value[:, 0, :, :, :]
        elif rank is 3:
            value = tf.expand_dims(value, axis=-1)
        perm = [0, 2, 1, 3]
        prepared_summary = tf.summary.image(sum_name, tf.transpose(value, perm), max_outputs=1)
    elif sum_type is "Category":
        if rank is not 4:
            ValueError("Time-series data not yet implemented for this type of summary")
        perm = [0, 3, 1, 2]
        prepared_summary = tf.summary.image(sum_name, tf.transpose(value, perm), max_outputs=1)
    elif sum_type is "Scalar":
        prepared_summary = tf.summary.scalar(sum_name, value)
    elif sum_type is "Histogram":
        prepared_summary = tf.summary.histogram(sum_name, value)

    return prepared_summary


def tb_setup_BasicFFClassifier(tb_logs_dir, input_data, model_output, target_output, gradients, loss, accuracy,
                               percent_positive, session):
    summaries = []

    # Loss summary
    summaries.append(prepare_summary(loss, "Loss", "Scalar"))
    summaries.append(prepare_summary(accuracy, "Accuracy", "Scalar"))
    summaries.append(prepare_summary(percent_positive, "Percent Positive", "Scalar"))

    # Gradient summary
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            summaries.append(prepare_summary(grad, var.name + '/grad', "Histogram"))

    # IO summary
    summaries.append(prepare_summary(input_data, "IO/Input", "Image"))
    summaries.append(prepare_summary(model_output, "IO/Output", "Category"))
    summaries.append(prepare_summary(target_output, "IO/Target", "Category"))

    # Create summarize op
    summarize_op = tf.summary.merge(summaries)
    summarizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

    # Open browser
    subprocess.call('fuser 6006/tcp -k &', shell=True)
    subprocess.call('tensorboard --logdir=' + tb_logs_dir + ' --reload_interval=10' + ' &', shell=True)
    ff = webbrowser.get('firefox')
    ff.open_new("http://127.0.1.1:6006/")

    return summarize_op, summarizer, summaries


def tb_flush(tb_logs_dir):
    shutil.rmtree(tb_logs_dir)


def display_to_console(i, session_results, summarizer, summarize, summarize_op, losses, accs):
    summarizer.add_summary(summarize_op, i)

    if summarize:
        llprint("\n\tAvg. Logistic Loss: %.4f\n" % (np.mean(losses)))
        llprint("\n\tAvg. Accuracy: %.4f\n" % (np.mean(accs)))
        losses = []
        accs = []

    return losses, accs


def take_checkpoint(ncomputer, session, ckpts_dir, task, iterations):
    llprint("\nSaving Checkpoint ... "), \
    ncomputer.save(session, os.path.join(ckpts_dir, task), 'step-%d' % (iterations))
    llprint("Done!\n")


def softmax_safe(X, axis=[0], eps=1e-6):
    num = tf.exp(X)
    den = tf.reduce_sum(num, axis=axis)

    for a in axis:
        den = tf.expand_dims(den, axis=a)
    sm = tf.div(num, den + eps)
    return sm


def bad_catcher(rv, g, var_names):
    rv_inf_catcher = {str(k): [] for k in range(len(rv))}
    rv_nan_catcher = {str(k): [] for k in range(len(rv))}
    g_nan_catcher = {var_names[k]: [] for k in range(len(g))}
    g_inf_catcher = {var_names[k]: [] for k in range(len(g))}

    rv_nan_names = []
    rv_inf_names = []
    g_nan_names = []
    g_inf_names = []

    # Find RV NaNs and Infs
    for ff in range(len(rv)):
        the_nans = np.argwhere(np.isnan(rv[ff]))
        the_infs = np.argwhere(np.isinf(rv[ff]))
        if len(the_nans) > 0:
            rv_nan_catcher[str(ff)].append(the_nans)
            rv_nan_names.append(str(ff))
        if len(the_infs) > 0:
            rv_inf_catcher[str(ff)].append(the_infs)
            rv_inf_names.append(str(ff))

    # Find G NaNs and Infs
    for ff in range(len(g)):
        the_nans = np.argwhere(np.isnan(g[ff]))
        the_infs = np.argwhere(np.isinf(g[ff]))
        if len(the_nans) > 0:
            g_nan_catcher[var_names[ff]].append(the_nans)
            g_nan_names.append(var_names[ff])
        if len(the_infs) > 0:
            g_inf_catcher[var_names[ff]].append(the_infs)
            g_inf_names.append(var_names[ff])

    return [rv_nan_catcher, rv_inf_catcher, g_nan_catcher, g_inf_catcher, rv_nan_names, rv_inf_names, g_nan_names,
            g_inf_names]

def rgb2grey(rgb):
    return(np.dot(rgb[...,:3], [0.299, 0.587, 0.114]))

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


def get_patch_loc(img_dims, item_size, img):
    img = np.squeeze(img)
    img[np.where(img == 255)] = 1.
    r = img_dims[0] - item_size[0] + 1
    c = img_dims[1] - item_size[0] + 1
    iter_o = np.array(list(product(range(0,item_size[0]), range(0,item_size[1]))))
    left_corners = reduce(lambda x,y: np.multiply(x,y), map(lambda x, y: img[x:x+r, y:y+c], iter_o[:, 0], iter_o[:, 1]))
    left_top_locations = np.transpose(np.where(left_corners > 0))
    left_top_locations = np.array(left_top_locations, dtype=np.int32)
    return left_top_locations

def sep_boxes(img_dims, itm_size, nitems, labels, img_type_lbl):
    left_mask = []
    right_mask = []
    mid_point = (img_dims[0]/2, img_dims[1]/2)
    for index in range(0, len(labels)):
        lbl_x = labels[index:index + 1, :]
        img_type_x = img_type_lbl[index]
        # img_key_x = img_key[index]

        top_left_loc = get_patch_loc(img_dims, itm_size, lbl_x)
        quad_dict = determine_quad(top_left_loc, mid_point)

        mask_1 = np.zeros(img_dims[0]*img_dims[1])
        mask_2 = np.zeros(img_dims[0]*img_dims[1])
        mask_1 = np.reshape(mask_1, (img_dims[0], img_dims[1]))
        mask_2 = np.reshape(mask_2, (img_dims[0], img_dims[1]))

        if 'left_box' in quad_dict:
            coords = quad_dict['left_box']
            mask_1[coords[0]: coords[0] + itm_size[0], coords[1]: coords[1] + itm_size[0]] = 1.
            coords = quad_dict['right_box']
            mask_2[coords[0]:coords[0] + itm_size[0], coords[1]: coords[1] + itm_size[0]] = 1.
        elif 'top_box' in quad_dict:
            coords = quad_dict['top_box']
            mask_1[coords[0]: coords[0] + itm_size[0], coords[1]: coords[1] + itm_size[0]] = 1.
            coords = quad_dict['bottom_box']
            mask_2[coords[0]:coords[0] + itm_size[0], coords[1]: coords[1] + itm_size[0]] = 1.

        left_mask.append(np.squeeze(mask_1))
        right_mask.append(np.squeeze(mask_2))
        # plt.imsave(img_dir + str(img_key_x) + "/" + str(img_type_x) + '/labels/mask_patch_1.png',
        #            np.squeeze(mask_1))
        # plt.imsave(img_dir + str(img_key_x) + "/" + str(img_type_x) + '/labels/mask_patch_2.png',
        #            np.squeeze(mask_2))

    return np.array(left_mask), np.array(right_mask)



def check_position_viability(new_position, old_position, SR_label, SR_portion, item_size, SR_type):
    # FIRST BITPATTERN POSITION
    y_pos_1 = new_position[0]
    x_pos_1 = new_position[1]

    # SECOND BITPATTERN POSITION
    y_pos_2 = old_position[0]
    x_pos_2 = old_position[1]

    # RESAMPLE IF NECESSARY
    viability = 1
    if SR_type == 'average_orientation' or SR_type == 'average_displacement':
        # print('check position viability:', new_position, old_position)
        if ((np.abs(y_pos_1 - y_pos_2) <= item_size[0]) & (np.abs(x_pos_1 - x_pos_2) <= item_size[1])):
            viability = 0
    elif SR_type == 'all':
        if SR_label == 0:  # LEFT-RIGHT
            if (np.abs(x_pos_2 - x_pos_1) <= (SR_portion * np.float(np.abs(y_pos_2 - y_pos_1)))) | \
                    ((np.abs(y_pos_1 - y_pos_2) <= item_size[0]) & (np.abs(x_pos_1 - x_pos_2) <= item_size[
                        1])):  # if horizontal distance is LEQ THAN sp_portion X vertical distance
                viability = 0
        elif SR_label == 1:  # UP-DOWN
            if (np.abs(x_pos_2 - x_pos_1) > (SR_portion * np.float(np.abs(y_pos_2 - y_pos_1)))) | \
                    ((np.abs(y_pos_1 - y_pos_2) <= item_size[0]) & (np.abs(x_pos_1 - x_pos_2) <= item_size[
                        1])):  # while horizontal distance is GREATER     THAN sp_portion X vertical distance
                viability = 0
        elif SR_label is None:  # SR AGNOSTIC
            if (np.abs(y_pos_1 - y_pos_2) <= item_size[0]) & (np.abs(x_pos_1 - x_pos_2) <= item_size[1]):
                viability = 0
        else:
            raise ValueError('SR_label should be 0 or 1 or None')

    return viability
