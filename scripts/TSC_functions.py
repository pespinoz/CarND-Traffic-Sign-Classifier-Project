from tensorflow.contrib.layers import flatten
import pickle
import numpy as np
import cv2
import random
import tensorflow as tf
from sklearn.utils import shuffle
import os
import matplotlib.pyplot as plt
#%matplotlib inline # Visualizations will be shown in the notebook.
plt.interactive(True)

with open('../saved/trained_model.pickle', 'rb') as f:
    accuracy_training, accuracy_validation, n_classes, TRAINING_FILE, VALIDATION_FILE, TESTING_FILE, \
    SIGNNAMES_FILE, EPOCHS, BATCH_SIZE, MU, SIGMA, RATE, CHANNELS, SAVED_MODEL, DATASET_DISPLAY, \
    SCREENSIZE, KEEP_PROB = pickle.load(f)

###############################################################################################################

class TimeExecution():
    from datetime import datetime
    def __enter__(self):
        self.start = self.datetime.now()
    def __exit__(self, *args, **kwargs):
        print('Runtime: {}'.format(self.datetime.now() - self.start))

def read_sign_names(file):
    fid = open(file, 'r')
    lines = fid.readlines()
    t = 0
    number = []
    name = []
    for line in lines:
        aux = line.split(',')
        if t >= 1:
            number.append(aux[0])
            name.append(aux[1])
        t += 1
    fid.close()
    return number, name


def preprocessing_grayscale(image_data):
    return cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()


def preprocessing_normalization(image_data):
    xmin = np.min(image_data)
    xmax = np.max(image_data)
    a = -0.5
    b = 0.5
    return a + (image_data-xmin)*(b-a)/(xmax-xmin)


def conv2d(input, filter_shape, filter_mean, filter_stddev, strides, padding_scheme, activation):
    # Filter weight shape: (height, width, input_depth, output_depth) / Filter bias shape: (output_depth)
    # Stride shape: (batch_size, height, width, depth)
    f_w = tf.Variable(tf.truncated_normal(shape=filter_shape, mean=filter_mean, stddev=filter_stddev))
    f_b = tf.Variable(tf.zeros([filter_shape[3]]))
    conv = tf.nn.conv2d(input, f_w, strides, padding_scheme) + f_b
    if activation == 'YES':
        conv = tf.nn.relu(conv)
    return conv
# Output shape for the 'VALID' padding:
# out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
# out_width  = ceil(float(in_width  - filter_width  + 1) / float(strides[2]))
# out_depth  = filter_depth (filter_shape[3])
# Output shape for the 'SAME' padding:
# out_height = ceil(float(in_height) / float(strides[1]))
# out_width  = ceil(float(in_width) / float(strides[2]))
# out_depth  = filter_depth (filter_shape[3])


def maxpool(input, filter_shape, strides, padding_scheme):
    return tf.nn.max_pool(input, ksize=filter_shape, strides=strides, padding=padding_scheme)
# Output shape for the 'VALID' padding:
# out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
# out_width  = ceil(float(in_width  - filter_width  + 1) / float(strides[2]))
# out_depth  = input depth
# Output shape for the 'SAME' padding:
# out_height = ceil(float(in_height) / float(strides[1]))
# out_width  = ceil(float(in_width) / float(strides[2]))
# out_depth  = input depth

def fully_connected(input, filter_shape, filter_mean, filter_stddev, activation):
    f_w = tf.Variable(tf.truncated_normal(shape=filter_shape, mean=filter_mean, stddev=filter_stddev))
    f_b = tf.Variable(tf.zeros(filter_shape[1]))
    fc = tf.matmul(input, f_w) + f_b
    if activation == 'YES':
        fc = tf.nn.relu(fc)
    return fc
# Output shape:
# out_depth  = filter width (filter_shape[1])


def model_architecture(x, keep_prob):
    layer1 = conv2d(x, [3, 3, CHANNELS, 12], MU, SIGMA, [1, 1, 1, 1], 'VALID', 'YES') # 32x32xch to 30x30x12
    layer1 = maxpool(layer1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')                     # 30x30x12 to 15x15x12

    layer2 = conv2d(layer1, [3, 3, 12, 32], MU, SIGMA, [1, 1, 1, 1], 'VALID', 'YES')  # 15x15x12 to 13x13x32
    layer2 = maxpool(layer2, [1, 2, 2, 1], [1, 1, 1, 1], 'VALID')                     # 13x13x32 to 12x12x32

    layer3 = conv2d(layer2, [3, 3, 32, 64], MU, SIGMA, [1, 1, 1, 1], 'VALID', 'YES')  # 12x12x32 to 10x10x64
    layer4 = conv2d(layer3, [3, 3, 64, 128], MU, SIGMA, [1, 1, 1, 1], 'VALID', 'YES') # 10x10x64 to 8x8x128
    layer4 = maxpool(layer4, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')                     # 8x8x128  to 4x4x128

    layer5 = fully_connected(flatten(layer4), [2048, 512], MU, SIGMA, 'YES')          # 4x4x128  to 2048 to 512
    layer5 = tf.nn.dropout(layer5, keep_prob)
    layer6 = fully_connected(layer5, [512, 128], MU, SIGMA, 'YES')                    # 512      to 128
    layer6 = tf.nn.dropout(layer6, keep_prob)
    layer7 = fully_connected(layer6, [128, n_classes], MU, SIGMA, 'NO')               # 128      to n_classes
    return layer7


def load_images_from_folder(folder):
    new_images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            new_images.append(img)
    return new_images
