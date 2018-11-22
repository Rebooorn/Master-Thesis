from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf

'''
These are all helper functions used to architect the network

reference: 
https://github.com/jakeret/tf_unet
'''


def weight_variable(shape, std_dev = 0.1, name="weight"):
    tmp = tf.truncated_normal(shape=shape,
                              stddev=std_dev)
    return tf.Variable(initial_value=tmp,
                       name=name)


def bias_variable(shape, initial = 0.1, name='bias'):
    tmp = tf.constant(initial, shape=shape)
    return tf.Variable(initial_value=tmp,
                       name=name)


def conv2d(input, weight, bias, prob):
    '''
    helper function to add 2D conv layer,
    default no padding and stride

    :param input: input tensor
    :param weight: weight of conv layer
    :param bias: bias of conv layer
    :param prob: keep probability for drop out
    :return: output tensor
    '''
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(input=input,
                               filter=weight,
                               strides=[1,1,1,1],
                               padding='VALID')
        conv_2d = tf.nn.bias_add(value=conv_2d,
                                 bias=bias)

        # drop-out
        drop_out = tf.nn.dropout(x=conv_2d,
                                 keep_prob=prob)
        return drop_out


def transposed_conv2d(input, weight, stride):
    '''
    Helper function for 2D transposed conv layer
    by default, the scaling size is [*1, *2, *2, *1/2]

    :param input:
    :param weight:
    :param stride:
    :return:
    '''
    with tf.name_scope("transposed_conv2d"):
        x_shape = tf.shape(input=input)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        transconv2d = tf.nn.conv2d_transpose(value=input,
                                             filter=weight,
                                             output_shape=output_shape,
                                             strides=[1, stride, stride, 1],
                                             padding='VALID')
        return transconv2d


def max_pool(input, nsize):
    '''
    Helper function to architect maxpooling layer,

    :param input:
    :param nsize:
    :return:
    '''
    return tf.nn.max_pool(input,
                          ksize=[1, nsize, nsize, 1],
                          strides=[1, nsize, nsize, 1],
                          padding='VALID')


def crop_and_concat(x1, x2):
    '''
    crop x1 to fit the size of x2
    :param x1: input tensor, on down-stream
    :param x2: up-stream tensor,
    :return: concatenation of x1_cropped and x2
    '''
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offset for the top left corner of the crop
        start = [0, (x1_shape[1] - x2_shape[1])//2, (x1_shape[2] - x2_shape[2])//2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, start, size)
        return tf.concat([x1_crop, x2], axis=3)


def pixelwise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exp_map = tf.exp(output_map - max_axis)
        normalized_map = tf.reduce_sum(exp_map, axis=3, keepdims=True)
        return exp_map / normalized_map


def cross_entropy(label, output_map):
    return -tf.reduce_mean()

