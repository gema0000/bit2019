#  실습 : tf14_mnist_dropout.py에 아래를 적용해서 리파인하시오.

import tensorflow as tf

# W1 = tf.get_variable("W1", shape=[?, ?],
#                      initializer=tf.random_uniform_initializer())
# b1 = tf.Variable(tf.random_normal([512]))
# L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

tf.constant_initializer()
tf.zeros_initializer()
tf.random_uniform_initializer()
tf.random_normal_initializer()
tf.contrib.layers.xavier_initializer()  # 킹왕짱
