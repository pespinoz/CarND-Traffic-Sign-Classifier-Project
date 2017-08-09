from TSC_functions import *
with open('../saved/trained_model.pickle', 'rb') as f:
    accuracy_training, accuracy_validation, n_classes, TRAINING_FILE, VALIDATION_FILE, TESTING_FILE, \
    SIGNNAMES_FILE, EPOCHS, BATCH_SIZE, MU, SIGMA, RATE, CHANNELS, SAVED_MODEL, DATASET_DISPLAY, \
    SCREENSIZE, KEEP_PROB = pickle.load(f)
NEW_IMS_DIR = '../german_web_images/set_all'

################################################################################################################

# opening and preprocessing test images
with open(TESTING_FILE, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']
X_test_original = X_test

X_test = np.array(list(map(lambda x: preprocessing_grayscale(x), X_test_original)))
X_test = X_test[..., None]
X_test = np.array(list(map(lambda x: preprocessing_normalization(x), X_test)))

#restoring TF and plotting (and printing) accuracy
graph = tf.Graph()
with graph.as_default():
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

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
        tf.train.Saver().restore(sess, SAVED_MODEL)
        print("TensorFlow Model Restored")
        print()

        def evaluate_accuracy(X_data, y_data):
            num_examples = len(X_data)
            total_accuracy = 0
            sess = tf.get_default_session()
            for offset in range(0, num_examples, BATCH_SIZE):
                batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
                accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                total_accuracy += (accuracy * len(batch_x))
            return total_accuracy / num_examples

        accuracy_test = evaluate_accuracy(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(accuracy_test))
        print()

        fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=SCREENSIZE)
        # ax1, ax2 = axes.ravel()
        line1, = ax1.plot(range(EPOCHS), accuracy_training, 'r-', linewidth=2, label='Training')
        line2, = ax1.plot(range(EPOCHS), accuracy_validation, 'b-', linewidth=2, label='Validation')
        ax1.legend(loc='lower right', frameon=False)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        # line1, = ax2.plot(range(EPOCHS), loss_training, 'r-', linewidth=2, label='Training')
        # line2, = ax2.plot(range(EPOCHS), loss_validation, 'b-', linewidth=2, label='Validation')
        # ax2.legend(loc='lower right', frameon=False)
        # ax2.set_xlabel('Epochs')
        # ax2.set_ylabel('Loss')
        # fig.savefig('../my_images/accuracy_loss.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', \
        #  papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)
        #plt.close(fig)

''' Step 3: Test a Model on New Images
To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
You may find signnames.csv useful as it contains mappings from the class id (integer) to the actual sign name.'''

'''Load and Output the Images'''
### Load the images and plot them here. ### Feel free to use as many code cells as needed.


def load_images_from_folder(folder):
    new_images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            new_images.append(img)
    return new_images

new_images = np.array(load_images_from_folder(NEW_IMS_DIR))

X_new_original = np.empty((len(os.listdir(NEW_IMS_DIR)), 32, 32, 3), dtype=np.uint8)
for i in range(X_new_original.shape[0]):
    X_new_original[i, :, :, :] = cv2.resize(new_images[i], (32, 32), interpolation=cv2.INTER_AREA)
    X_new_original[i, :, :, :] = cv2.cvtColor(X_new_original[i, :, :, :], cv2.COLOR_BGR2RGB)

X_new = np.empty((len(os.listdir(NEW_IMS_DIR)), 32, 32, CHANNELS), dtype=np.uint8)
if CHANNELS == 1:
    X_new = np.array(list(map(lambda x: preprocessing_grayscale(x), X_new_original)))
    X_new = X_new[..., None]
    X_new = np.array(list(map(lambda x: preprocessing_normalization(x), X_new)))
else:
    assert CHANNELS == 3
    for channel in range(CHANNELS):
        X_new[:, :, :, channel] = np.array(list(map(lambda x: cv2.equalizeHist(x), X_new_original[:, :, :, channel])))

fig = plt.figure(figsize=(SCREENSIZE[0], SCREENSIZE[1]/2))
fig.suptitle('Top row: Raw. Bottom row: preprocessed', fontsize=12)
for j in range(X_new.shape[0]):
    plt.subplot(2, X_new.shape[0], j+1)
    plt.axis('off')
    plt.imshow(X_new_original[j, :, :])
    if CHANNELS == 1:
        plt.subplot(2, X_new.shape[0], X_new.shape[0] + j + 1)
        plt.axis('off')
        plt.imshow(np.squeeze(X_new[j, :, :]), cmap='gray')
    if CHANNELS == 3:
        plt.subplot(2, X_new.shape[0], X_new.shape[0] + j + 1)
        plt.axis('off')
        plt.imshow(X_new[j, :, :])
    fig.savefig('../my_images/new_images.jpg', dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None,
                format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)
#plt.close(fig)


'''Predict the Sign Type for Each Image'''
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

graph = tf.Graph()
with graph.as_default():
    xnew = tf.placeholder(tf.float32, (None, 32, 32, CHANNELS))
    keep_prob = tf.placeholder(tf.float32)
    logits = model_architecture(xnew, keep_prob)

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
        tf.train.Saver().restore(sess, SAVED_MODEL)
        print("TensorFlow Model Restored")
        print()

        predicted_logits = sess.run(logits, feed_dict={xnew: X_new, keep_prob: 1.0})
        predicted_labels = np.argmax(predicted_logits, axis=1)

        sign_number, sign_name = read_sign_names(SIGNNAMES_FILE)
        fig = plt.figure(figsize=(SCREENSIZE[0], SCREENSIZE[1] / 2))
        for j in range(6):
            plt.subplot(2, len(range(6)), j + 1)
            plt.axis('off')
            plt.imshow(X_new_original[j, :, :])
            plt.subplot(2, len(range(6)), len(range(6)) + j + 1)
            plt.axis('off')
            plt.text(0.5, 1.05, sign_name[predicted_labels[j]], horizontalalignment='center', verticalalignment = 'center', fontsize=9)
        fig.savefig('../my_images/new_images_with_labels_part1.jpg', dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                    pad_inches=0.1, frameon=None)
        #plt.close(fig)
        fig = plt.figure(figsize=(SCREENSIZE[0], SCREENSIZE[1] / 2))
        t=0
        for j in range(6, X_new.shape[0]):
            plt.subplot(2, len(range(6)), t + 1)
            plt.axis('off')
            plt.imshow(X_new_original[j, :, :])
            plt.subplot(2, len(range(6)), len(range(6)) + t + 1)
            plt.axis('off')
            plt.text(0.5, 1.05, sign_name[predicted_labels[j]], horizontalalignment='center', verticalalignment = 'center', fontsize=9)
            t+=1
        fig.savefig('../my_images/new_images_with_labels_part2.jpg', dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                    pad_inches=0.1, frameon=None)
        #plt.close(fig)

'''Analyze Performance'''
### Calculate the accuracy for these 5 new images.
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


'''Output Top 5 Softmax Probabilities For Each Image Found on the Web'''

graph = tf.Graph()
with graph.as_default():
    xnew = tf.placeholder(tf.float32, (None, 32, 32, CHANNELS))
    keep_prob = tf.placeholder(tf.float32)
    logits = model_architecture(xnew, keep_prob)
    softmax_logits = tf.nn.softmax(logits)
    top_probs = tf.nn.top_k(logits, 5)

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
        tf.train.Saver().restore(sess, SAVED_MODEL)
        print("TensorFlow Model Restored")
        print()

        top_probabilities = sess.run(top_probs, feed_dict={xnew: X_new, keep_prob: 1.0})

        sign_number, sign_name = read_sign_names(SIGNNAMES_FILE)
        fig = plt.figure(figsize=(SCREENSIZE[0], SCREENSIZE[1] / 2))
        for j in range(6):
            plt.subplot(2, len(range(6)), j + 1)
            plt.axis('off')
            plt.imshow(X_new_original[j, :, :])
            plt.subplot(2, len(range(6)), len(range(6)) + j + 1)
            plt.axis('off')
            plt.text(0.5, 1.0, sign_name[top_probabilities[1][j][0]], horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.95, "({:.2f}%)".format(top_probabilities[0][j][0]), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.8, sign_name[top_probabilities[1][j][1]], horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.75, "({:.2f}%)".format(top_probabilities[0][j][1]), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.6, sign_name[top_probabilities[1][j][2]], horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.55, "({:.2f}%)".format(top_probabilities[0][j][2]), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.4, sign_name[top_probabilities[1][j][3]], horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.35, "({:.2f}%)".format(top_probabilities[0][j][3]), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.2, sign_name[top_probabilities[1][j][4]], horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.15, "({:.2f}%)".format(top_probabilities[0][j][4]), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
        fig.savefig('../my_images/new_images_with_toplabels_part1.jpg', dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                        pad_inches=0.1, frameon=None)
        # plt.close(fig)
        fig = plt.figure(figsize=(SCREENSIZE[0], SCREENSIZE[1] / 2))
        t=0
        for j in range(6, X_new.shape[0]):
            plt.subplot(2, len(range(6)), t + 1)
            plt.axis('off')
            plt.imshow(X_new_original[j, :, :])
            plt.subplot(2, len(range(6)), len(range(6)) + t + 1)
            plt.axis('off')
            plt.text(0.5, 1.0, sign_name[top_probabilities[1][j][0]], horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.95, "({:.2f}%)".format(top_probabilities[0][j][0]), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.8, sign_name[top_probabilities[1][j][1]], horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.75, "({:.2f}%)".format(top_probabilities[0][j][1]), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.6, sign_name[top_probabilities[1][j][2]], horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.55, "({:.2f}%)".format(top_probabilities[0][j][2]), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.4, sign_name[top_probabilities[1][j][3]], horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.35, "({:.2f}%)".format(top_probabilities[0][j][3]), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.2, sign_name[top_probabilities[1][j][4]], horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            plt.text(0.5, 0.15, "({:.2f}%)".format(top_probabilities[0][j][4]), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)
            t+=1
        fig.savefig('../my_images/new_images_with_toplabels_part2.jpg', dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
                        pad_inches=0.1, frameon=None)
        # plt.close(fig)