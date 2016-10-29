from numpy import *
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import Image
import random as r
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from scipy.misc import imread
from scipy.misc import imresize


import tensorflow as tf
#sess = tf.InteractiveSession()

#Training Samples Dimensions

train_x = zeros((1,227, 227, 3)).astype(float32)
train_y = zeros((1, 5))

x_dim = train_x.shape[1:]
y_dim = train_y.shape[1]

#Initialize Weights
def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bais_variables(shape):
    initial = tf.constant( 0.1, shape= shape)
    return tf.Variable(initial)

#Convolution

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):	
    #From https://github.com/ethereon/caffe-tensorflow
    
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

#Max-pooling 
def max_pool(x, k_h, k_w, s_h, s_w, padding):
  return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding= padding)

# Local response normalization
def lrn(x, radius, alpha, beta, bias):
	return tf.nn.local_response_normalization( x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)



# Alex-Net Model


x_dummy = (random.random((1,)+ x_dim)/255.).astype(float32)
i = x_dummy.copy()
#x = tf.Variable(train_x)

x = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
x_image = tf.reshape(x, [-1, 227, 227, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 5])

#Covolution 1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1_W = weight_variables([11, 11, 3, 96])
conv1_b = bais_variables([96])
#conv1_in = conv(x_image, conv1_W, conv1_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1_in = tf.nn.conv2d(i, conv1_W, [1, s_h, s_w, 1], padding="SAME")
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = lrn(conv1, radius, alpha, beta, bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = max_pool(lrn1, k_h, k_w, s_h, s_w, padding)

#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2_W = weight_variables([5, 5, 48, 256])
conv2_b = bais_variables([256])
conv2_in = conv(maxpool1, conv2_W, conv2_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)

#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = lrn(conv2, radius, alpha, beta, bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = max_pool(lrn2, k_h, k_w, s_h, s_w,padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3_W = weight_variables([3, 3, 256, 384])
conv3_b = bais_variables([384])
conv3_in = conv(maxpool2, conv3_W, conv3_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4_W = weight_variables([3, 3, 192, 384])
conv4_b = bais_variables([384])
conv4_in = conv(conv3, conv4_W, conv4_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5_W = weight_variables([3, 3, 192, 256])
conv5_b = bais_variables([256])
conv5_in = conv(conv4, conv5_W, conv5_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)


#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = max_pool(conv5, k_h, k_w, s_h, s_w, padding)

print maxpool5.get_shape()
# Fully Connected Layers

#fc6
#fc(4096, name='fc6')
fc6_W = weight_variables([9216, 4096])
fc6_b = bais_variables([4096])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [1, int(prod(maxpool5.get_shape()[1:]))]), fc6_W, fc6_b)



#fc7
#fc(4096, name='fc7')
fc7_W = weight_variables([4096, 4096])
fc7_b = bais_variables([4096])
fc7 = tf.nn.relu(tf.nn.matmul(fc6, fc7_W) + fc7_b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8_W = weight_variables([4096, 1000])
fc8_b = bais_variables([1000])
fc8 = tf.nn.relu(fc7, fc8_W, fc8_b)

#Dropout
keep_prob = tf.placeholder(tf.float32)
fc8_drop = tf.nn.dropout( fc8, keep_prob)

#Readout Layer
W_fc = weight_variables([1000, 5])
b_fc = bais_variables([5])

y_conv=tf.nn.softmax(tf.matmul(fc8_drop, W_fc) + b_fc)




########################################################
# Input 

dataset_path      = "/home/somnath/tf_alexnet/data"
test_labels_file  = "label_test.txt"
train_labels_file = "label_train.txt"

test_set_size = 200

IMAGE_HEIGHT  = 227
IMAGE_WIDTH   = 227
NUM_CHANNELS  = 3
BATCH_SIZE    = 50

def encode_label(label):
    return int(label)

def read_label_file(file):
    f = open(file, "r")
    filepaths = []
    labels = []
    for line in f:
        filepath, label_str = line.split(";")
        label = [int(x) for x in label_str.split(',')]
        filepaths.append(filepath)
        labels.append((label))
    return filepaths, labels

# reading labels and file path
train_filepaths, train_labels = read_label_file('label_train.txt')
test_filepaths, test_labels = read_label_file('label_test.txt')

# transform relative path into full path
train_path = "/train/"
test_path = "/test/"
train_filepaths = [ dataset_path + train_path + fp[0:9] + "/" + fp for fp in train_filepaths]
test_filepaths = [ dataset_path + test_path + fp[0:9] + "/" + fp for fp in test_filepaths]

# for this example we will create or own test partition
all_filepaths = train_filepaths + test_filepaths
all_labels = train_labels + test_labels

# convert string into tensors
all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

# create a partition vector
partitions = [0] * len(all_filepaths)
partitions[:test_set_size] = [1] * test_set_size
r.shuffle(partitions)

# partition our data into a test and train set according to our partition vector
train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

# create input queues
train_input_queue = tf.train.slice_input_producer([train_images, train_labels], shuffle=True)

test_input_queue = tf.train.slice_input_producer([test_images, test_labels], shuffle=True)

# process path and string tensor into an image and a label
file_content = tf.read_file(train_input_queue[0])
image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
train_image = tf.image.resize_images(image, 227, 227)
train_label = train_input_queue[1]

file_content = tf.read_file(test_input_queue[0])
image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
test_image = tf.image.resize_images(image, 227, 227)
test_label = test_input_queue[1]

# define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


# collect batches of images before processing
train_image_batch, train_label_batch = tf.train.batch([train_image, train_label], batch_size=BATCH_SIZE #,num_threads=1
                                                      )
test_image_batch, test_label_batch = tf.train.batch([test_image, test_label], batch_size=BATCH_SIZE #,num_threads=1
                                                    )

print train_label_batch.get_shape()
print y_.get_shape()
print "input pipeline ready"


################################################################################
#Train


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    for i in range(20000):
        train_batch_image = sess.run(train_image_batch)
        train_batch_label = sess.run(train_label_batch)

        #if i%100 == 0:
        train_accuracy = accuracy.eval(session = sess, feed_dict={x: train_batch_image, y_: train_batch_label, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(session=sess, feed_dict={x: train_batch_image, y_: train_batch_label, keep_prob: 0.5})

    test_batch_image = sess.run(train_image_batch)
    test_batch_label = sess.run(train_label_batch)

    print("test accuracy %g "%accuracy.eval(session = sess, feed_dict={x: test_batch_image, y_: test_batch_label, keep_prob: 1.0}))



    coord.request_stop()
    coord.join(threads)
    sess.close()

