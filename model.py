from numpy import *
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
import sort_data
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

def bias_variables(shape):
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
    return  conv

#Max-pooling 
def max_pool(x, k_h, k_w, s_h, s_w, padding):
  return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding= padding)

# Local response normalization
def lrn(x, radius, alpha, beta, bias):
	return tf.nn.local_response_normalization( x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)



# Alex-Net Model

def alexnet(x_image, keep_prob):
    
    #Covolution 1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1_W = weight_variables([11, 11, 3, 96])
    conv1_b = bias_variables([96])
    conv1_in = conv(x_image, conv1_W, conv1_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    #conv1_in = tf.nn.conv2d(x_image, conv1_W, [1, s_h, s_w, 1], padding="SAME")
    conv1 = tf.nn.relu(conv1_in + conv1_b)

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
    conv2_b = bias_variables([256])
    conv2_in = conv(maxpool1, conv2_W, conv2_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    #conv2_in = tf.nn.conv2d(maxpool1, conv2_W, [1, s_h, s_w, 1], padding="SAME")
    conv2 = tf.nn.relu(conv2_in + conv2_b)

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
    conv3_b = bias_variables([384])
    conv3_in = conv(maxpool2, conv3_W, conv3_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    #conv3_in = tf.nn.conv2d(maxpool2, conv3_W, [1, s_h, s_w, 1], padding="SAME")
    conv3 = tf.nn.relu(conv3_in + conv3_b)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4_W = weight_variables([3, 3, 192, 384])
    conv4_b = bias_variables([384])
    conv4_in = conv(conv3, conv4_W, conv4_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    #conv4_in = tf.nn.conv2d(conv3, conv4_W, [1, s_h, s_w, 1], padding="SAME")
    conv4 = tf.nn.relu(conv4_in + conv4_b)


    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5_W = weight_variables([3, 3, 192, 256])
    conv5_b = bias_variables([256])
    conv5_in = conv(conv4, conv5_W, conv5_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    #conv5_in = tf.nn.conv2d(conv4, conv5_W, [1, s_h, s_w, 1], padding="SAME")
    conv5 = tf.nn.relu(conv5_in + conv5_b)


    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = max_pool(conv5, k_h, k_w, s_h, s_w, padding)
    print maxpool5.get_shape()

    # Fully Connected Layers

    #fc6
    #fc(4096, name='fc6')
    fc6_W = weight_variables([9216, 4096])
    fc6_b = bias_variables([4096])
    maxpool5_flat = tf.reshape(maxpool5, [-1, 9216])
    fc6 = tf.nn.relu(tf.matmul( maxpool5_flat, fc6_W) + fc6_b)
    
    #fc7
    #fc(4096, name='fc7')
    fc7_W = weight_variables([4096, 2048])
    fc7_b = bias_variables([2048])
    fc7 = tf.nn.relu(tf.matmul(fc6, fc7_W) + fc7_b)
    
    
    #fc8
    #fc(2048, relu=False, name='fc8')
    fc8_W = weight_variables([2048, 2048])
    fc8_b = bias_variables([2048])
    fc8 = tf.nn.relu(tf.matmul(fc7, fc8_W) + fc8_b)
    
    #fc9
    #fc(1024, relu=False, name='fc8')
    fc9_W = weight_variables([2048, 1024])
    fc9_b = bias_variables([1024])
    fc9 = tf.nn.relu(tf.matmul(fc8, fc9_W) + fc9_b)
    
    #Dropout
    fc8_drop = tf.nn.dropout( fc9, keep_prob)

    #Readout Layer
    W_fc = weight_variables([1024, 3])
    b_fc = bias_variables([3])

    y_conv=tf.nn.softmax(tf.matmul(fc8_drop, W_fc) + b_fc)

    return y_conv