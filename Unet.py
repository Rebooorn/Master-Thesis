import tensorflow as tf
import numpy as np
import logging
import Layers

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def _get_image_summary(input_tensor):
    '''
    here extract image summary from 4D tensor
    '''
    V = tf.slice(input_tensor, (0,0,0,0), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    w = tf.shape(input_tensor)[1]
    h = tf.shape(input_tensor)[2]
    V = tf.reshape(V, (w, h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, w, h, 1))
    return V


def create_unet(input, keep_prob, channels, n_class, layers = 5, feature_root=32, filter_size=3, pool_size=2,
                summary = True):
    '''
    u_net is created here.
    :param input: input batch (or placeholder)
    :param keep_prob: used in drop-out layer
    :return: output tensor,
    '''
    logging.info('U-net details: ')
    logging.info('encoding depth: {depth}'.format(
        depth=layers
    ))

    # some preprocessing
    with tf.name_scope('reshape_beforehand'):
        nx = tf.shape(input)[1]
        ny = tf.shape(input)[2]
        input_image = tf.reshape(input, [-1, nx, ny, channels])
        in_node = input_image
        batch_size = tf.shape(input_image)[0]

    in_size = 572
    size = in_size
    dw_maps = []
    image_summary = []  # list of summary images
    feature = feature_root

    # path down
    for layer in range(layers):
        with tf.name_scope('down_conv_{:0>2d}'.format(layer)):
            feature *= 2
            if layer == 0:
                w1 = Layers.weight_variable(shape=[filter_size, filter_size, channels, feature],
                                            name='w1')
            else:
                w1 = Layers.weight_variable(shape=[filter_size, filter_size, feature//2, feature],
                                            name='w1')
            b1 = Layers.bias_variable(shape=[feature], name='b1')
            w2 = Layers.weight_variable(shape=[filter_size, filter_size, feature, feature],
                                        name='w2')
            b2 = Layers.bias_variable(shape=[feature], name='b2')

            # two concatenated conv
            conv1 = Layers.conv2d(in_node, w1, b1, keep_prob)
            relu_conv1 = tf.nn.relu(conv1)
            conv2 = Layers.conv2d(relu_conv1, w2, b2, keep_prob)
            relu_conv2 = tf.nn.relu(conv2)

            dw_maps.append(relu_conv2)
            image_summary.append(conv2)

            size -= 4
            in_node = relu_conv2
            if layer < layers - 1:
                # max-pooling
                pool = Layers.max_pool(relu_conv2, pool_size)
                in_node = pool
                size /= 2

    # up path
    for layer in range(-1, -layers, -1):
        with tf.name_scope('up_conv_{:0>2d}'.format(abs(layer))):
            feature = feature // 2
            size *= 2
            # up conv layer
            wd = Layers.weight_variable(shape=[pool_size, pool_size, feature, feature*2],
                                        name='wd')
            bd = Layers.bias_variable(shape=[feature], name='bd')
            trans_conv = Layers.transposed_conv2d(input=in_node,
                                                  weight=wd,
                                                  bias=bd,
                                                  stride=pool_size)
            # crop and concatenation
            crop_concat = Layers.crop_and_concat(dw_maps.pop(), trans_conv)
            # conv1
            w1 = Layers.weight_variable(shape=[filter_size, filter_size, feature*2, feature],
                                        name='w1')
            b1 = Layers.bias_variable(shape=[feature], name='b1')
            conv1 = Layers.conv2d(input=crop_concat,
                                  weight=w1,
                                  bias=b1,
                                  prob=keep_prob)
            relu_conv1 = tf.nn.relu(conv1)
            # conv2
            w2 = Layers.weight_variable(shape=[filter_size, filter_size, feature, feature],
                                        name='w2')
            b2 = Layers.bias_variable(shape=[feature], name='b2')
            conv2=Layers.conv2d(input=relu_conv1,
                                weight=w2,
                                bias=b2,
                                prob=keep_prob)

            image_summary.append(conv2)
            relu_conv2 = tf.nn.relu(conv2)
            size -= 4
            in_node = relu_conv2

    # 1*1 conv to output segmentation map
    with tf.name_scope('output_map'):
        wo = Layers.weight_variable(shape=[1, 1, feature, n_class], name='wo')
        bo = Layers.bias_variable([2], name='bo')
        output = Layers.conv2d(input=in_node,
                               weight=wo,
                               bias=bo,
                               prob=1.0)
        output = tf.nn.relu(output)

    if summary:
        # generate the summary of images
        with tf.name_scope('summarites'):
            # more summaries possible here
            for i, conv in enumerate(image_summary):
                tf.summary.image('summary_conv_{:0>2d}'.format(i), _get_image_summary(conv))

    return output

class unet:
    def __init__(self, channels=3, n_class=2):
        '''
        An initial implementation of
        '''
        # here construct the architecture
        self.input_tensor = input_


