import tensorflow as tf
# print(tf.__version__)

hello = tf.constant("Hello Yellow")

sess = tf.Session()

print(sess.run(hello))

