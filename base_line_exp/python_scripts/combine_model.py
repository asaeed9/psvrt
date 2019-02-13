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
from gen_data_batch import generate_batch, generate_batch_1


def see_output_grey(iNp, depth_filter_to_see=0, cmap="gray", figsize=(4, 4)):
    img_x = iNp[0, :, :]
    fig = plt.figure(figsize=figsize)
    plt.imshow(img_x, interpolation='none', aspect='auto')
    #     plt.colorbar(img_x, orientation='horizontal')
    plt.show()


def see_output(iNp, depth_filter_to_see=0, cmap="gray", figsize=(4, 4)):
    img_x = iNp[0, :, :, :]
    fig = plt.figure(figsize=figsize)
    if cmap == "gray":
        plt.imshow(img_x, cmap=plt.get_cmap('gray'))
    else:
        plt.imshow(img_x, interpolation='none', aspect='auto')
    # plt.colorbar(img_x, orientation='horizontal')
    plt.show()


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


def new_weights(shape, layer_name):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape), name=layer_name + '_W')


def new_bias(length, layer_name):
    return tf.Variable(tf.constant(0.05, shape=[length]), name=layer_name + '_b')


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

        layer = tf.add(tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME'), biases,
                       name=layer_name)
        #         print('layer:', layer)
        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME', name=layer_name + '_max')
        layer = tf.nn.relu(layer, name=layer_name + '_activation')

    # print('maxpooled layer:', layer)

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

        layer = tf.add(tf.matmul(input, weights), biases, name=layer_name)
        #     layer = tf.matmul(input, weights) + biases

        if use_relu:
            layer = tf.nn.relu(layer, layer_name + '_activation')

    return layer


def argmax_2d(tensor, bat_len):
    inp_tensor_shape = tf.shape(tensor)
    nimgs = inp_tensor_shape[0]
    img_dims = inp_tensor_shape[1:]
    img_len = inp_tensor_shape[-1]
    flat_img = tf.reshape(tensor, [-1, tf.reduce_prod(img_dims)])

    # # argmax of the flat tensor
    argmax_x = tf.map_fn(lambda img: tf.cast(tf.argmax(img), tf.float64) // tf.cast(img_len, dtype=tf.float64),
                         tf.cast(flat_img, dtype=tf.float64))
    argmax_y = tf.map_fn(lambda img: tf.cast(tf.argmax(img), tf.float64) % tf.cast(img_len, dtype=tf.float64),
                         tf.cast(flat_img, dtype=tf.float64))

    res = tf.stack([argmax_x, argmax_y, tf.zeros([bat_len], dtype=tf.float64)], axis=1)

    return res


# def extract_patch(orig_vec, mask_vec):
#     batch_len = tf.shape(mask_vec)[0]
#     input_shape = (img_shape[0], img_shape[1])
#     rows, cols = input_shape[0], input_shape[1]
#     item_shape = tf.Variable([item_size[0], item_size[1]])
#     d_rows, d_cols = item_shape[0], item_shape[1]
#     subm_rows, subm_cols = rows - d_rows + 1, cols - d_cols + 1

#     ii, jj = tf.meshgrid(tf.range(d_rows), tf.range(d_cols), indexing='ij')
#     d_ii, d_jj = tf.meshgrid(tf.range(subm_rows), tf.range(subm_cols), indexing='ij')

#     subm_ii = ii[:subm_rows, :subm_cols, tf.newaxis, tf.newaxis] + d_ii
#     subm_jj = jj[:subm_rows, :subm_cols, tf.newaxis, tf.newaxis] + d_jj

#     subm_st = tf.stack([subm_ii, subm_jj], axis=-1)

#     gather_exp = tf.map_fn(lambda mask: tf.gather_nd(mask, subm_st), mask_vec)

#     subm_dims = tf.shape(gather_exp)
#     gather_exp = tf.reshape(gather_exp, [-1,  subm_dims[1] * subm_dims[2], subm_dims[3], subm_dims[4]])
#     reduced_mat = tf.map_fn(lambda mask: tf.scan(lambda a, b: tf.multiply(a, b), tf.squeeze(mask))[-1], gather_exp)
#     pred_crds = argmax_2d(reduced_mat, batch_len)

#     pred_crds = tf.cast(pred_crds, dtype = tf.int64)

#     itms = tf.map_fn(lambda idx: tf.cast(tf.slice(orig_vec[tf.cast(idx, tf.int64), :, :, :],pred_crds[tf.cast(idx, tf.int64), :], [item_size[0],item_size[1], 3]), dtype = tf.float64), tf.cast(tf.range(batch_len), dtype = tf.float64))
# #     itms = tf.map_fn(lambda idx: tf.cast(tf.slice(tf.squeeze(vec[idx:idx+1, :, :]), tf.squeeze(pred_crds[idx:idx+1, :]), [item_size[0], item_size[1]]), dtype=tf.int32), tf.range(batch_len))

#     return itms


def extract_patch(orig_vec, mask_vec):
    batch_len = tf.shape(mask_vec)[0]
    input_shape = (img_shape[0], img_shape[1])
    rows, cols = input_shape[0], input_shape[1]
    item_shape = tf.Variable([item_size[0], item_size[1]])
    d_rows, d_cols = item_shape[0], item_shape[1]
    subm_rows, subm_cols = rows - d_rows + 1, cols - d_cols + 1

    ii, jj = tf.meshgrid(tf.range(d_rows), tf.range(d_cols), indexing='ij')
    d_ii, d_jj = tf.meshgrid(tf.range(subm_rows), tf.range(subm_cols), indexing='ij')

    subm_ii = ii[:subm_rows, :subm_cols, tf.newaxis, tf.newaxis] + d_ii
    subm_jj = jj[:subm_rows, :subm_cols, tf.newaxis, tf.newaxis] + d_jj

    subm_st = tf.stack([subm_ii, subm_jj], axis=-1)

    gather_exp = tf.map_fn(lambda mask: tf.gather_nd(mask, subm_st), mask_vec)

    subm_dims = tf.shape(gather_exp)
    gather_exp = tf.reshape(gather_exp, [-1, subm_dims[1] * subm_dims[2], subm_dims[3], subm_dims[4]])
    reduced_mat = tf.map_fn(lambda mask: tf.scan(lambda a, b: tf.multiply(a, b), tf.squeeze(mask))[-1], gather_exp)
    pred_crds = argmax_2d(reduced_mat, batch_len)

    pred_crds = tf.cast(pred_crds, dtype=tf.int64, name="predicted_crds")

    itms = tf.map_fn(lambda idx: tf.cast(
        tf.slice(orig_vec[tf.cast(idx, tf.int64), :, :, :], pred_crds[tf.cast(idx, tf.int64), :],
                 [item_size[0], item_size[1], 3]), dtype=tf.float64), tf.cast(tf.range(batch_len), dtype=tf.float64))
    #     itms = tf.map_fn(lambda idx: tf.cast(tf.slice(tf.squeeze(vec[idx:idx+1, :, :]), tf.squeeze(pred_crds[idx:idx+1, :]), [item_size[0], item_size[1]]), dtype=tf.int32), tf.range(batch_len))

    return itms


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
    return result[0], result[1]


def test_mask_sd(batch_size, model_mask_sd):
    np.set_printoptions(suppress=True)
    sd_accuracy = 0.
    mask_accuracy = 0.
    sig_mask_accuracy = 0.

    test_set, mask_test_labels, test_labels = generate_batch_1(batch_size, img_shape, item_size, nitems)

    mask_logits, sd = restore_see_layer(orig=test_set, input_name='x', model_name=model_mask_sd,
                                        var_name=['train_mask/mask_6/fc3', 'mask_sd/mask_conv_layer/sd_y_pred'])
    #     mask_logits, pred_mask, sd = restore_see_layer(orig=test_set, input_name = 'x', model_name=model_mask_sd, var_name=['train_mask/mask_6/fc3', 'train_mask/sigmoid_output', 'train_mask/original_items/TensorArrayStack/TensorArrayGatherV3'])

    #     mask_logits[mask_logits < 0 ] = 0

    #     print('Predicted Mask Logits: ', mask_logits[0].shape)
    #     print('Predicted Mask: ', pred_mask.shape)
    #     print('Original Mask: ', mask_test_labels)
    #     print('Test SD: ', test_set)

    sd_accuracy = np.sum(np.argmax(sd, axis=1) == np.argmax(test_labels, axis=1)) / batch_size

    #     cor_prd_imgs = np.sum([True for pm, tm  in zip(pred_mask, mask_test_labels) if np.allclose(pm, tm)])
    #     sig_cor_prd_imgs = np.sum([True for pm, tm  in zip(pred_mask, mask_test_labels) if np.allclose(pm, tm)])

    #     print(cor_prd_imgs)
    #     mask_accuracy = cor_prd_imgs/batch_size
    #     sig_mask_accuracy = sig_cor_prd_imgs/batch_size

    #     print('Mask Test Accuracy:{0:>.4f}, Sigmoid Test Accuracy:{1:>.4f}'.format(mask_accuracy, sig_mask_accuracy))

    print('Mask Test Accuracy:{0:>.4f}, SD Test Accuracy:{1:>.4f}'.format(mask_accuracy, sd_accuracy))

    return mask_accuracy, sd_accuracy, mask_logits, mask_test_labels


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def plotNNFilter(units):
    print(units.shape)
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest")


def getActivations(layer, feed_dict_train):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        units = session.run(layer, feed_dict=feed_dict_train)

        plotNNFilter(units)

single_patch_model = "SD/complete_model.ckpt"
total_imgs = 20000
train_batch_size = 128
img_shape = (60,60,3)
item_size = (5,5)
nitems = 2
num_classes = 2

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, img_shape[0] * img_shape[1] * img_shape[2]], name='x')
x_image = tf.reshape(x, [-1, img_shape[0], img_shape[1], img_shape[2]])

mask_y_true = tf.placeholder(tf.float32, shape=[None, img_shape[0] * img_shape[1]], name='mask_y_true')
mask_y_true_cls = tf.argmax(mask_y_true, axis=1)

sd_y_true = tf.placeholder(tf.float32, shape=[None, 2], name='sd_y_true')
sd_y_true_cls = tf.argmax(sd_y_true, axis=1)

initializer = tf.contrib.layers.xavier_initializer()

name_scope = 'train_mask'

# mask_graph layer configurations
# mask_graph layer configurations
m_filter_size0 = 6  # Convolution filters(kernel) are 4 x 4 pixels.
m_num_filters0 = 10  # There are 16 of these filters.

m_filter_size1 = 6  # Convolution filters are 4 x 4 pixels.
m_num_filters1 = 10  # There are 16 of these filters.

# Convolutional Layer 2.
m_filter_size2 = 4  # Convolution filters are 2 x 2 pixels.
m_num_filters2 = 16  # There are 16 of these filters.

m_filter_size3 = 4  # Convolution filters are 2 x 2 pixels.
m_num_filters3 = 8  # There are 4 of these filters.

# Convolutional Layer 3.
m_filter_size4 = 2  # Convolution filters are 2 x 2 pixels.
m_num_filters4 = 8  # There are 32 of these filters.

m_filter_size5 = 2  # Convolution filters are 2 x 2 pixels.
m_num_filters5 = 8  # There are 16 of these filters.

# Fully-connected layer.
m_fc_size = 2000  # Number of neurons in fully-connected layer.

"""SD Network Layer Architecture"""
filter_size1 = 4  # Convolution filters are 4 x 4 pixels.
num_filters1 = 16  # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 2  # Convolution filters are 2 x 2 pixels.
num_filters2 = 32  # There are 32 of these filters.

# Convolutional Layer 3.
filter_size3 = 2  # Convolution filters are 2 x 2 pixels.
num_filters3 = 64  # There are 64 of these filters.

# Convolutional Layer 4.
filter_size4 = 2  # Convolution filters are 2 x 2 pixels.
num_filters4 = 128  # There are 128 of these filters.

# Fully-connected layer.
fc_size = 256  # Number of neurons in fully-connected layer.

with tf.name_scope(name_scope):
    # First Convolution Layer
    layer_name = 'mask_conv_layer'
    shape = [m_filter_size0, m_filter_size0, img_shape[2], m_num_filters0]
    mask_weights = tf.Variable(initializer(shape), name=layer_name + '_W')
    mask_biases = tf.Variable(tf.constant(0.05, shape=[m_num_filters0]), name=layer_name + '_b')

    layer0_conv0, weights_conv0 = new_conv_layer(input=x_image,
                                                 num_input_channels=img_shape[2],
                                                 filter_size=m_filter_size0,
                                                 num_filters=m_num_filters0,
                                                 name_scope='mask',
                                                 layer_name='conv1',
                                                 use_pooling=False)
    #     layer0_conv0_drpout = tf.nn.dropout(layer0_conv0, 0.3, name="drop_out")

    layer1_conv1, weights_conv1 = new_conv_layer(input=layer0_conv0,
                                                 num_input_channels=m_num_filters0,
                                                 filter_size=m_filter_size1,
                                                 num_filters=m_num_filters1,
                                                 name_scope='mask',
                                                 layer_name='conv2',
                                                 use_pooling=False)

    #     layer1_conv1_drpout = tf.nn.dropout(layer1_conv1, 0.3, name="drop_out")

    layer2_conv2, weights_conv2 = new_conv_layer(input=layer1_conv1,
                                                 num_input_channels=m_num_filters1,
                                                 filter_size=m_filter_size2,
                                                 num_filters=m_num_filters2,
                                                 name_scope='mask',
                                                 layer_name='conv3',
                                                 use_pooling=True)

    layer3_conv3, weights_conv3 = new_conv_layer(input=layer2_conv2,
                                                 num_input_channels=m_num_filters2,
                                                 filter_size=m_filter_size3,
                                                 num_filters=m_num_filters3,
                                                 name_scope='mask',
                                                 layer_name='conv4',
                                                 use_pooling=False)

    layer4_conv4, weights_conv4 = new_conv_layer(input=layer3_conv3,
                                                 num_input_channels=m_num_filters3,
                                                 filter_size=m_filter_size4,
                                                 num_filters=m_num_filters4,
                                                 name_scope='mask',
                                                 layer_name='conv5',
                                                 use_pooling=False)

    layer5_conv5, weights_conv5 = new_conv_layer(input=layer4_conv4,
                                                 num_input_channels=m_num_filters4,
                                                 filter_size=m_filter_size5,
                                                 num_filters=m_num_filters5,
                                                 name_scope='mask',
                                                 layer_name='conv6',
                                                 use_pooling=False)

    #     layer6_conv6, weights_conv6 =  new_conv_layer(input=layer5_conv5,
    #                                            num_input_channels=m_num_filters5,
    #                                            filter_size=m_filter_size6,
    #                                            num_filters=m_num_filters6,
    #                                              name_scope = 'mask',
    #                                              layer_name = 'conv7',
    #                                            use_pooling=True)

    layer_flat, num_features = flatten_layer(layer5_conv5)
    #     print(layer_flat)
    #     layer_fc1 = new_fc_layer(input=layer_flat,
    #                              num_inputs=num_features,
    #                              num_outputs=m_fc_size,
    #                              name_scope = 'mask',
    #                              layer_name = 'fc1',
    #                              use_relu=False)
    #     print('layer_fc1:', layer_fc1)
    #     layer_fc2 = new_fc_layer(input=layer_fc1,
    #                              num_inputs=m_fc_size,
    #                              num_outputs=m_fc_size,
    #                              name_scope = 'mask',
    #                              layer_name = 'fc2',
    #                              use_relu=False)

    #     layer_fc3 = new_fc_layer(input=layer_fc2,
    #                              num_inputs=m_fc_size,
    #                              num_outputs=m_fc_size,
    #                              name_scope = 'mask',
    #                              layer_name = 'fc3',
    #                              use_relu=False)
    #     mask_drop_out = tf.nn.dropout(layer_fc3, 0.5, name="drop_out")

    mask_layer_fc3 = new_fc_layer(input=layer_flat,
                                  num_inputs=num_features,
                                  num_outputs=img_shape[0] * img_shape[1],
                                  name_scope='mask',
                                  layer_name='fc3',
                                  use_relu=False)

    #     mask_drop_out = tf.nn.dropout(mask_layer_fc4, 0.5, name="drop_out")
    #     y_pred = tf.nn.softmax(mask_drop_out, name="softmax_output")
    y_pred = mask_layer_fc3

    # round numbers less than 0.5 to zero;
    # by making them negative and taking the maximum with 0
    #     differentiable_round = tf.maximum(y_pred-0.499,0)
    # scale the remaining numbers (0 to 0.5) to greater than 1
    # the other half (zeros) is not affected by multiplication
    #     differentiable_round = differentiable_round * 10000
    # take the minimum with 1
    #     y_pred_round = tf.minimum(differentiable_round, 1)
    #     print(differentiable_round)

    y_pred_sigmoid = tf.nn.sigmoid(y_pred, name="sigmoid_output")
    y_pred_sigmoid = tf.reshape(y_pred_sigmoid, [-1, img_shape[0], img_shape[1]])

    batch_len = tf.shape(y_pred_sigmoid)[0]
    input_shape = (img_shape[0], img_shape[1])
    rows, cols = input_shape[0], input_shape[1]
    item_shape = tf.Variable([item_size[0], item_size[1]])
    d_rows, d_cols = item_shape[0], item_shape[1]
    subm_rows, subm_cols = rows - d_rows + 1, cols - d_cols + 1

    ii, jj = tf.meshgrid(tf.range(d_rows), tf.range(d_cols), indexing='ij')
    d_ii, d_jj = tf.meshgrid(tf.range(subm_rows), tf.range(subm_cols), indexing='ij')

    subm_ii = ii[:subm_rows, :subm_cols, tf.newaxis, tf.newaxis] + d_ii
    subm_jj = jj[:subm_rows, :subm_cols, tf.newaxis, tf.newaxis] + d_jj

    subm_st = tf.stack([subm_ii, subm_jj], axis=-1)

    gather_exp = tf.map_fn(lambda mask: tf.gather_nd(mask, subm_st), y_pred_sigmoid)

    subm_dims = tf.shape(gather_exp)
    gather_exp = tf.reshape(gather_exp, [-1, subm_dims[1] * subm_dims[2], subm_dims[3], subm_dims[4]])
    reduced_mat = tf.map_fn(lambda mask: tf.scan(lambda a, b: tf.multiply(a, b), tf.squeeze(mask))[-1], gather_exp)

    inp_tensor_shape = tf.shape(reduced_mat)
    nimgs = inp_tensor_shape[0]
    img_dims = inp_tensor_shape[1:]
    img_len = inp_tensor_shape[-1]
    flat_img = tf.reshape(reduced_mat, [-1, tf.reduce_prod(img_dims)])

    _, top_idx = tf.nn.top_k(flat_img, nitems)
    topx_idx = tf.squeeze(tf.cast(tf.reshape(top_idx, [-1, 1]), tf.float32) // tf.cast(img_len, tf.float32))
    topy_idx = tf.squeeze(tf.cast(tf.reshape(top_idx, [-1, 1]), tf.float32) % tf.cast(img_len, tf.float32))

    res12 = tf.stack([topx_idx, topy_idx, tf.zeros([batch_len * nitems], dtype=tf.float32)], axis=1, name="pred_crds12")

    pred_crds12 = tf.cast(res12, dtype=tf.int32, name="predicted_crds12")
    pred_crds = tf.map_fn(lambda idx: pred_crds12[idx: idx + 2], tf.range(0, batch_len * nitems, nitems),
                          dtype=tf.int32)

    items = tf.map_fn(lambda img_idx: tf.reshape(tf.map_fn(lambda itm_idx: tf.slice(x_image[img_idx, :, :, :],
                                                                                    pred_crds[img_idx, itm_idx, :],
                                                                                    [item_size[0], item_size[1], 3])
                                                           , tf.range(nitems), dtype=tf.float32),
                                                 [item_size[0] * nitems, item_size[0], 3])
                      , tf.range(batch_len), dtype=tf.float32)

    mask_loss = tf.square(mask_y_true - y_pred)
    mask_cost = tf.reduce_mean(mask_loss)

    #     train_op1 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(mask_loss)

    ### some more performance measures
    mask_correct_prediction = tf.equal(y_pred, mask_y_true)
    mask_accuracy = tf.reduce_mean(tf.cast(mask_correct_prediction, tf.float32))

name_scope = 'mask_sd'
with tf.name_scope(name_scope):
    input_sd = tf.cast(items, dtype=tf.float32, name=layer_name + "/input_sd")

    layer1_conv1, weights_conv1 = new_conv_layer(input=input_sd,
                                                 num_input_channels=img_shape[2],
                                                 filter_size=filter_size1,
                                                 num_filters=num_filters1,
                                                 name_scope='mask_sd_graph',
                                                 layer_name='conv1',
                                                 use_pooling=True)

    layer2_conv2, weights_conv2 = new_conv_layer(input=layer1_conv1,
                                                 num_input_channels=num_filters1,
                                                 filter_size=filter_size2,
                                                 num_filters=num_filters2,
                                                 name_scope='mask_sd_graph',
                                                 layer_name='conv2',
                                                 use_pooling=True)

    layer3_conv3, weights_conv3 = new_conv_layer(input=layer2_conv2,
                                                 num_input_channels=num_filters2,
                                                 filter_size=filter_size3,
                                                 num_filters=num_filters3,
                                                 name_scope='mask_sd_graph',
                                                 layer_name='conv3',
                                                 use_pooling=True)

    layer4_conv4, weights_conv4 = new_conv_layer(input=layer3_conv3,
                                                 num_input_channels=num_filters3,
                                                 filter_size=filter_size4,
                                                 num_filters=num_filters4,
                                                 name_scope='mask_sd_graph',
                                                 layer_name='conv4',
                                                 use_pooling=True)

    layer_flat, num_features = flatten_layer(layer4_conv4)

    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             name_scope='mask_sd_graph',
                             layer_name='fc1',
                             use_relu=True)

    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=fc_size,
                             num_outputs=fc_size,
                             name_scope='mask_sd_graph',
                             layer_name='fc2',
                             use_relu=False)

    layer_fc3 = new_fc_layer(input=layer_fc2,
                             num_inputs=fc_size,
                             num_outputs=fc_size,
                             name_scope='mask_sd_graph',
                             layer_name='fc3',
                             use_relu=False)

    drop_out = tf.nn.dropout(layer_fc3, 0.5)
    layer_fc4 = new_fc_layer(input=drop_out,
                             num_inputs=fc_size,
                             num_outputs=num_classes,
                             name_scope='mask_sd_graph',
                             layer_name='fc4',
                             use_relu=False)

    sd_y_pred = tf.nn.softmax(layer_fc4, name=layer_name + "/sd_y_pred")
    sd_y_pred_cls = tf.argmax(sd_y_pred, axis=1)

    sd_loss = tf.nn.softmax_cross_entropy_with_logits(logits=sd_y_pred, labels=sd_y_true)

    train_op1 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(mask_loss)
    train_op2 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(sd_loss)
    final_train_op = tf.group(train_op1, train_op2)

    sd_cost = tf.reduce_mean(sd_loss)

    sd_correct_prediction = tf.equal(sd_y_pred_cls, sd_y_true_cls)
    sd_accuracy = tf.reduce_mean(tf.cast(sd_correct_prediction, tf.float32))


def optimize(num_epochs, save_model=True, save_name="base_model", restore_model=False, restore_name=None):
    total_iterations = 0
    done_train_imgs = 0
    start_time = time.time()
    start_batch = 0
    end_batch = train_batch_size
    test_batch_size = 256
    plot_accuracy = []
    plot_accuracy_epoch = []
    plot_training_size = []
    plot_training_size_epoch = []
    plot_mask = []
    plot_mask_sd = []
    plot_sd = []
    plot_sr = []
    saver = tf.train.Saver()

    sum_accuracy = 0.0
    n = 1

    session = tf.Session()
    session.run(tf.global_variables_initializer())

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

            train_data, mask_labels, labels = generate_batch_1(train_batch_size, img_shape, item_size, nitems)
            #             print(train_data.shape)
            #             print(train_data[0])
            if not len(train_data) and not len(labels) and not len(mask_labels):
                print("All images have been processed.")
                break;

            feed_dict_train = {x: train_data,
                               mask_y_true: mask_labels,
                               sd_y_true: labels}

            #             feed_dict_train = {x: train_data,
            #                        mask_y_true: mask_labels}

            #             session.run(train_op1, feed_dict=feed_dict_train)

            session.run(final_train_op, feed_dict=feed_dict_train)
            mask_acc, mask_co, sd_acc, sd_co = session.run([mask_accuracy, mask_cost, sd_correct_prediction, sd_cost],
                                                           feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Mask Training Accuracy: {1:>6.1%}, Mask Loss: {2:>.4f}, SD Training Accuracy: {1:>6.1%}, SD Loss: {2:>.4f}"
            print(msg.format(end_batch + 1, mask_acc, mask_co, sd_acc, sd_co))

            #             mask_acc, mask_co, itms = session.run([mask_accuracy, mask_cost, items], feed_dict=feed_dict_train)
            #             msg = "Optimization Iteration: {0:>6}, Mask Training Accuracy: {1:>6.1%}, Mask Loss: {2:>.4f}"
            #             print(msg.format(end_batch + 1, mask_acc, mask_co))

            #             print(list(train_data[0]))
            #             print('Predicted Cords: ', list(pred_crd))

            #             fig=plt.figure(figsize=(8, 8))
            #             columns = 8
            #             rows = 8
            #             for i in range(1, columns*rows):
            #                 img = itms[i]
            #                 fig.add_subplot(rows, columns, i)
            #                 plt.imshow(img)
            #             plt.show()

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
        plot_mask.append(mask_acc)
        plot_sd.append(sd_acc)
        e_time = time.time()
        time_dif = e_time - s_time
        print("Test Mask SD Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    end_time = time.time()
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    print('plot mask accuracy:')
    print(plot_mask)
    print(plot_sd)

if __name__ == "__main__":

    save_model = True
    save_name = single_patch_model
    restore_model = False
    restore_name = single_patch_model
    optimize(num_epochs=20, save_model=True, save_name=single_patch_model, restore_model=restore_model,
             restore_name=single_patch_model)



