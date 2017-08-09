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

TRAINING_FILE = '../../traffic-signs-data/train.p'
VALIDATION_FILE = '../../traffic-signs-data/valid.p'
TESTING_FILE = '../../traffic-signs-data/test.p'
SIGNNAMES_FILE = '../signnames.csv'
EPOCHS = 10
BATCH_SIZE = 512
MU = 0
SIGMA = 0.1
RATE = 0.0025
CHANNELS = 1
SAVED_MODEL = '../saved/trained_model'
DATASET_DISPLAY = 5
SCREENSIZE = (16, 8.5)
KEEP_PROB = 0.7

class TimeExecution():
    from datetime import datetime
    def __enter__(self):
        self.start = self.datetime.now()
    def __exit__(self, *args, **kwargs):
        print('Runtime: {}'.format(self.datetime.now() - self.start))

################################################################################################################

''' Step 0: Load the Data '''

# Load pickled data
# TODO: Fill this in based on where you saved the training and testing data

with open(TRAINING_FILE, mode='rb') as f:
    train = pickle.load(f)
with open(VALIDATION_FILE, mode='rb') as f:
    valid = pickle.load(f)
with open(TESTING_FILE, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

X_train_original = X_train
X_valid_original = X_valid
X_test_original = X_test


''' Step 1: Dataset Summary & Exploration 
Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas '''
# Replace each question mark with the appropriate value.
# Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(np.concatenate((y_train, y_valid, y_test), axis=0)))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

''' Include an exploratory visualization of the dataset '''
# Data exploration visualization code goes here.
# Feel free to use as many code cells as needed.


def read_sign_names(file): # SIGNNAMES_FILE
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

sign_number, sign_name = read_sign_names(SIGNNAMES_FILE)

sign_counts = np.bincount(y_train)
index = []
for i in range(n_classes):
    #fig = plt.figure(figsize=(SCREENSIZE[0], SCREENSIZE[1]/2))
    #fig.suptitle('ClassID: ' + str(sign_number[i]) + '. ' + str(sign_name[i]) + 'Ocurrences in Training Dataset: ' + str(sign_counts[i]), fontsize=12)
    for j in range(DATASET_DISPLAY):
    #    plt.subplot(1, DATASET_DISPLAY, j+1)
    #    plt.axis('off')
        index.append(random.randint(0, np.nonzero(y_train == i)[0].shape[0]-1))
    #    plt.imshow(X_train_original[np.nonzero(y_train == i)[0][index[-1]], :, :])
    #fig.savefig('../my_images/visualization'+str(i)+'_ch'+str(CHANNELS)+'.jpg', dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None,
    #            format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)
    #plt.close(fig)

'''fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=SCREENSIZE)
colors = ['red', 'tan', 'lime']
ax0.hist([y_train, y_valid, y_test], n_classes, normed=1, histtype='bar', color=colors, label=['Training', 'Validation', 'Test'])
ax0.legend(prop={'size': 10}, loc='upper right', frameon=False)
ax0.set_title('Distribution of Classes in the Training, Validation and Test Set')
ax0.set_xlabel('Classes')
ax0.set_ylabel('Ocurrences (normalized)')
fig.savefig('../my_images/classes_distribution.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)
plt.close(fig)'''

''' Step 2: Design and Test a Model Architecture
Pre-process the Data Set (normalization, grayscale, etc.) '''


def preprocessing_grayscale(image_data):
    return cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()


def preprocessing_normalization(image_data):
    xmin = np.min(image_data)
    xmax = np.max(image_data)
    a = -0.5
    b = 0.5
    return a + (image_data-xmin)*(b-a)/(xmax-xmin)

if CHANNELS == 1:
    X_train = np.array(list(map(lambda x: preprocessing_grayscale(x), X_train_original)))
    X_train = X_train[..., None]
    X_train = np.array(list(map(lambda x: preprocessing_normalization(x), X_train)))

    X_valid = np.array(list(map(lambda x: preprocessing_grayscale(x), X_valid_original)))
    X_valid = X_valid[..., None]
    X_valid = np.array(list(map(lambda x: preprocessing_normalization(x), X_valid)))

    X_test = np.array(list(map(lambda x: preprocessing_grayscale(x), X_test_original)))
    X_test = X_test[..., None]
    X_test = np.array(list(map(lambda x: preprocessing_normalization(x), X_test)))
else:
    assert CHANNELS == 3
    for channel in range(CHANNELS):
        X_train[:, :, :, channel] = np.array(list(map(lambda x: cv2.equalizeHist(x), X_train_original[:, :, :, channel])))
        X_valid[:, :, :, channel] = np.array(list(map(lambda x: cv2.equalizeHist(x), X_valid_original[:, :, :, channel])))
        X_test[:, :, :, channel] = np.array(list(map(lambda x: cv2.equalizeHist(x), X_test_original[:, :, :, channel])))

'''t = 0
for i in range(n_classes):
    fig = plt.figure(figsize=(SCREENSIZE[0], SCREENSIZE[1]/2))
    fig.suptitle('ClassID: ' + str(sign_number[i]) + '. ' + str(sign_name[i]) + 'Preprocessed', fontsize=12)
    for j in range(DATASET_DISPLAY):
        plt.subplot(1, DATASET_DISPLAY, j+1)
        plt.axis('off')
        if CHANNELS == 1:
            plt.imshow(np.squeeze(X_train[np.nonzero(y_train == i)[0][index[t]], :, :]), cmap='gray')
            t += 1
        if CHANNELS == 3:
            plt.imshow(X_train[np.nonzero(y_train == i)[0][index[t]], :, :])
            t += 1
    fig.savefig('../my_images/preproc'+str(i)+'_ch'+str(CHANNELS)+'.jpg', dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None,
                format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)
    plt.close(fig)'''


''' Model Architecture '''


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


with TimeExecution():
    graph = tf.Graph()
    with graph.as_default():
        X_train, y_train = shuffle(X_train, y_train)
        x = tf.placeholder(tf.float32, (None, 32, 32, CHANNELS))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, n_classes)

        keep_prob = tf.placeholder(tf.float32)
        logits = model_architecture(x, keep_prob)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=RATE)
        training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        def evaluate_accuracy(X_data, y_data):
            num_examples = len(X_data)
            total_accuracy = 0
            sess = tf.get_default_session()
            for offset in range(0, num_examples, BATCH_SIZE):
                batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
                accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                total_accuracy += (accuracy * len(batch_x))
            return total_accuracy / num_examples

        print("Training...")
        print()

        accuracy_training = np.empty((0, EPOCHS))
        accuracy_validation = np.empty((0, EPOCHS))
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: KEEP_PROB})

            accuracy_training = np.append(accuracy_training, [evaluate_accuracy(X_train, y_train)])
            accuracy_validation = np.append(accuracy_validation, [evaluate_accuracy(X_valid, y_valid)])

            print("EPOCH {} ...".format(i + 1))
            print("Training Accuracy = {:.3f}".format(accuracy_training[i]), "Validation Accuracy = {:.3f}".format(accuracy_validation[i]))
            print()

        saver.save(sess, SAVED_MODEL)
        print("Model saved")


with open(SAVED_MODEL + '.pickle', 'wb') as f:
    pickle.dump([accuracy_training, accuracy_validation, n_classes, TRAINING_FILE, VALIDATION_FILE, TESTING_FILE,
                 SIGNNAMES_FILE, EPOCHS, BATCH_SIZE, MU, SIGMA, RATE, CHANNELS, SAVED_MODEL, DATASET_DISPLAY,
                 SCREENSIZE, KEEP_PROB], f)
