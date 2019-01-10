import logging
import time
# import tensorflow as tf
import sys, os
import numpy as np
import tensorflow as tf
from datetime import datetime

'''
path 
'''
print(__file__)
print(os.path.dirname(__file__))

'''
**kwarg test
'''
def test_kwarg(**kwargs):
    print(kwargs.get('name'))
    print(kwargs.get('age'))
    if kwargs.get('gender') == 'Male':
        print('this is a man')
    print('salary: ',kwargs.get('salary'))

test_kwarg(name='Chang', age=23, gender='female', work='master student')




'''
gpu accessing test
'''

# device_name = 'gpu'  # Choose device from cmd line. Options: gpu or cpu
# shape = (1500, 1500)
# if device_name == "gpu":
#     device_name = "/gpu:0"
# else:
#     device_name = "/cpu:0"
#
# with tf.device(device_name):
#     random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
#     dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
#     sum_operation = tf.reduce_sum(dot_operation)
#
#
# startTime = datetime.now()
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
#         result = session.run(sum_operation)
#         print(result)
#
# # It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
# print("\n" * 5)
# print("Shape:", shape, "Device:", device_name)
# print("Time taken:", datetime.now() - startTime)
#
# print("\n" * 5)

'''
this is the test for logging things
'''

# logging.basicConfig(filename='example.log', format='%(asctime)s %(message)s', level=logging.INFO)
#
# logging.info('new logging test start')
# for i in range(20):
#     logging.info('this is the {times} times of info logging'.format(times=i))
#     logging.info('haha \r\t haha')


'''
Tensorflow save and restore checkpoint
'''
# Create some variables.
# v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
# v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)
#
# inc_v1 = v1.assign(v1+1)
# dec_v2 = v2.assign(v2-1)
#
# # Add an op to initialize the variables.
# init_op = tf.global_variables_initializer()
#
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
#
# # Later, launch the model, initialize the variables, do some work, and save the
# # variables to disk.
# with tf.Session() as sess:
#     sess.run(init_op)
#     # Do some work with the model.
#     inc_v1.op.run()
#     dec_v2.op.run()
#     # Save the variables to disk.
#     save_path = saver.save(sess, "D:\\ChangLiu\\MasterThesis\\Master-Thesis\\model\\model.ckpt")
#     print("Model saved in path: %s" % save_path)

# restore
# tf.reset_default_graph()
# #
# # # Create some variables.
# # v1 = tf.get_variable("v1", shape=[3])
# # v2 = tf.get_variable("v2", shape=[5])
# #
# # # Add ops to save and restore all the variables.
# # saver = tf.train.Saver()
# #
# # # Later, launch the model, use the saver to restore variables from disk, and
# # # do some work with the model.
# # with tf.Session() as sess:
# #     # Restore variables from disk.
# #     saver.restore(sess, "D:\\ChangLiu\\MasterThesis\\Master-Thesis\\model\\model.ckpt")
# #     print("Model restored.")
# #     # Check the values of the variables
# #     print("v1 : %s" % v1.eval())
# #     print("v2 : %s" % v2.eval())

# tensorflow gpu test, whether gpu can be connected
