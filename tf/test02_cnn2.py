import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNISAT_data/", one_hot=True)

learning_rate = 0.001
training_epoch = 15
batch_size = 100

# X = tf.placeholder(tf.float32, [None,784])    # deprecated
X = tf.compat.v1.placeholder(tf.float32, [None,784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
# Y = tf.placeholder(tf.float32, [None, 10])    # deprecated
Y = tf.compat.v1.placeholder(tf.float32, [None, 10])
X2 = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])    # reshape할 필요는 없다.

print(X)
print(X2)
print(X_img)

# L1 ImgIn shape=(?, 28, 28, 1)
# W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))    # deprecated
W1 = tf.Variable(tf.random.normal([3, 3, 1, 32], stddev=0.01))
print('W1 : ', W1)
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
print('L1 : ', L1)
L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   # deprecated
L1 = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

print('L1 : ', L1)

'''
# L2 ImgIn shape=(?, 14, 14, 32)
# W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))   # deprecated
W2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
# L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   # deprecated
L2 = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])
'''
