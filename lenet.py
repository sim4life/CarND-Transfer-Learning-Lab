import os
import math
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from skimage import exposure
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=0)


# Number of training examples
n_train = len(X_train)

# Number of testing examples.
n_test = len(X_test)

# The shape of an traffic sign image
image_shape = X_train[0].shape

# Unique classes/labels in the dataset.
n_classes = max(y_train)+1

print("Number of training examples =", n_train)
print("Number of validation examples =", len(X_valid))
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

from skimage.exposure import rescale_intensity

def rgb2gray(imgs):
    # convert to grayscale
    return np.mean(imgs, axis=3, keepdims=True)

def normalize(imgs):
    # normalize to [-1, 1] range
    imgs_norm = (imgs - imgs.mean()) / (np.max(imgs) - np.min(imgs))
    return imgs_norm

def equalize(imgs):
    # equalize contrast
    new_imgs = np.empty(imgs.shape, dtype=float)
    for i, img in enumerate(imgs):
        new_imgs[i] = rescale_intensity(img)

    return new_imgs

def preprocess(imgs):
    new_imgs = equalize(imgs)
    new_imgs = rgb2gray(new_imgs)
    return new_imgs

# preprocess the images
X_train_processed = preprocess(X_train)
# X_valid_processed = preprocess(X_valid)
X_valid = preprocess(X_valid)
X_test = preprocess(X_test)

X_train, y_train = shuffle(X_train_processed, y_train)

print("New Image data shape =", X_train[0].shape)

# Parameters
# LEARNING_RATE = 0.001
EPOCHS = 15
BATCH_SIZE = 128

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
TEST_VALID_SIZE = 512

# Network Parameters
dropout = 0.75  # Dropout, probability to keep units
mu = 0
sigma = 0.1


# Store LeNet layers weight & bias
weights = {
    'wc1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma)),
    'wc2': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
    'wd1': tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma)),
    'wd2': tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma)),
    'out': tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))}

biases = {
    'bc1': tf.Variable(tf.random_normal([6])), # tf.zeros(6)
    'bc2': tf.Variable(tf.random_normal([16])), # tf.zeros(16)
    'bd1': tf.Variable(tf.random_normal([120])), # tf.zeros(120)
    'bd2': tf.Variable(tf.random_normal([84])), # tf.zeros(84)
    'out': tf.Variable(tf.random_normal([n_classes]))} # tf.zeros(n_classes)


def decay_learning_rate(size):
    divisor = 1000
    if size > 8:
        # divisor = 1000*size
        # divisor = math.e**size
        # divisor = 10*(size**math.e)
        divisor = size**math.e
    if size > 12:
        # divisor = math.e**(size-1)
        divisor = 10*(size**math.e)
    if size > 14:
        # divisor = math.e**(size-1)
        divisor = 20*(size**math.e)
    return 1/divisor

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def LeNet(x, weights, biases, dropout):

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # Layer 1 - 32*32*1 to 28*28*6 to 14*14*6
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Activation.
    conv1   = tf.nn.relu(conv1)
    # conv1 = tf.nn.dropout(conv1, dropout)
    # Pooling
    conv1 = maxpool2d(conv1, k=2)

    # Layer 2: Convolutional. Input = 14*14*6. Output = 5x5x16.
    # Layer 2 - 14*14*6 to 10*10*16 to 5*5*16
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Activation.
    conv2   = tf.nn.relu(conv2)
    # conv2 = tf.nn.dropout(conv2, (dropout*0.9))
    # Pooling
    conv2 = maxpool2d(conv2, k=2)

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = tf.add(tf.matmul(fc0, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

    return logits

# tf Graph input
x = tf.placeholder(tf.float32, [None, 32, 32, 1])
y = tf.placeholder(tf.int32, [None])
one_hot_y = tf.one_hot(y, n_classes)
keep_prob = tf.placeholder(tf.float32)
decaying_learning_rate = tf.placeholder(tf.float32) # 0.001

# Model
logits = LeNet(x, weights, biases, keep_prob)

# Define loss (cost) and optimizer (training_operation)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=decaying_learning_rate)
# model_save_dir = 'LeNet_model_sgd'
optimizer = tf.train.AdamOptimizer(learning_rate=decaying_learning_rate)
model_save_dir = 'LeNet_model_adam'
training_operation = optimizer.minimize(cost)

# Accuracy (accuracy_operation)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

saver = tf.train.Saver()

def evaluate_data(X_data, y_data):
    num_examples = len(X_data)
    total_loss, total_accuracy = 0, 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        # cost is loss_operation
        loss, accuracy = sess.run([cost, accuracy_op], feed_dict={x: batch_x, y: batch_y, keep_prob:1.})
        total_loss     += (loss * len(batch_x))
        total_accuracy += (accuracy * len(batch_x))
    return total_loss / num_examples, total_accuracy / num_examples


# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    num_examples = len(X_train)

    # steps_per_epoch = num_examples // BATCH_SIZE
    print("Training...")
    print()

    for epoch in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train) # to ensure training isn't biased by the order of images
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, decaying_learning_rate: decay_learning_rate(epoch+1)})

            '''
            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            valid_acc = sess.run(accuracy_op, feed_dict={
                x: X_validation[:TEST_VALID_SIZE],
                y: y_validation[:TEST_VALID_SIZE],
                keep_prob: 1.})

            print('Epoch {:>2}, Batch {:>3} - Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))


        val_loss, val_acc = eval_data(mnist.validation)
        print("EPOCH {} ...".format(epoch+1))
        print("Validation loss = {:.3f}".format(val_loss))
        print("Validation accuracy = {:.3f}".format(val_acc))
        print()
        '''

        validation_loss, validation_accuracy = evaluate_data(X_valid, y_valid)
        print("EPOCH {} ...".format(epoch+1))
        print("Validation Loss = {:.3f}".format(validation_loss))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()


    '''
    # Calculate Test Accuracy
    test_acc = sess.run(accuracy_op, feed_dict={
        x: mnist.test.images[:TEST_VALID_SIZE],
        y: mnist.test.labels[:TEST_VALID_SIZE],
        keep_prob: 1.})
    print('Testing Accuracy with prob: {}'.format(test_acc))

    # Evaluate on the test data
    test_loss, test_acc = eval_data(mnist.test)
    print("Eval_func Test loss = {:.3f}".format(test_loss))
    print("Eval_func Test accuracy = {:.3f}".format(test_acc))
    '''

    saver.save(sess, model_save_dir+'/lenet_new')
    print("Model saved")


def evaluate_test_data():
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_save_dir+'/'))

        test_loss, test_accuracy = evaluate_data(X_test, y_test)
        print("Test Loss = {:.3f}".format(test_loss))
        print("Test Accuracy = {:.3f}".format(test_accuracy))

# train_model()
# evaluate_test_data()
