
import tensorflow as tf
import random as r
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from scipy.misc import imread
from scipy.misc import imresize

dataset_path      = "/home/somnath/tf_alexnet/data"
test_labels_file  = "label_test.txt"
train_labels_file = "label_train.txt"

test_set_size = 584

IMAGE_HEIGHT  = 224
IMAGE_WIDTH   = 224
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


def Dataset():

        # reading labels and file path
    train_filepaths, train_labels = read_label_file('label_train.txt')
    test_filepaths, test_labels = read_label_file('label_test.txt')

    # transform relative path into full path
    train_path = "/train/"
    test_path = "/test/"

    train_filepaths = [ dataset_path + train_path + fp[0:4] + "/" + fp for fp in train_filepaths]
    test_filepaths = [ dataset_path + test_path + fp[0:4] + "/" + fp for fp in test_filepaths]

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
    train_image = tf.image.resize_images(image, 224, 224)
    train_label = train_input_queue[1]

    file_content = tf.read_file(test_input_queue[0])
    image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    test_image = tf.image.resize_images(image, 224, 224)
    test_label = test_input_queue[1]

    # define tensor shape
    train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


    # collect batches of images before processing
    train_image_batch, train_label_batch = tf.train.batch([train_image, train_label], batch_size=BATCH_SIZE #,num_threads=1
                                                          )
    test_image_batch, test_label_batch = tf.train.batch([test_image, test_label], batch_size=BATCH_SIZE #,num_threads=1
                                                        )

    print "input pipeline ready"

    return train_image_batch, test_image_batch, train_label_batch, test_label_batch