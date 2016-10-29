from numpy import *
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
import sort_data
import model
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

################################################################################
#Train

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
x_image = tf.reshape(x, [-1, 224, 224, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 3])
keep_prob = tf.placeholder(tf.float32)

y_conv = model.alexnet(x_image, keep_prob)

train_image_batch, test_image_batch, train_label_batch, test_label_batch = sort_data.Dataset()

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.MomentumOptimizer(0.001, 0.9, use_locking=False, name='Momentum', use_nesterov=True).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i= 0
    train_accuracy = 0
    while(train_accuracy < 0.99):
        train_batch_image = sess.run(train_image_batch)
        train_batch_label = sess.run(train_label_batch)
        #if i%100 == 0:
        train_accuracy = accuracy.eval(session = sess, feed_dict={x: train_batch_image, y_: train_batch_label, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(session=sess, feed_dict={x: train_batch_image, y_: train_batch_label, keep_prob: 0.5})
        if i%100 == 0:
            test_batch_image = sess.run(train_image_batch)
            test_batch_label = sess.run(train_label_batch)
            print("test accuracy %g "%accuracy.eval(session = sess, feed_dict={x: test_batch_image, y_: test_batch_label, keep_prob: 1.0}))

        i = i+1

    test_batch_image = sess.run(train_image_batch)
    test_batch_label = sess.run(train_label_batch)

    print("test accuracy %g "%accuracy.eval(session = sess, feed_dict={x: test_batch_image, y_: test_batch_label, keep_prob: 1.0}))



    coord.request_stop()
    coord.join(threads)
    sess.close()

