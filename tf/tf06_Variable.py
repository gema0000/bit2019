# 랜덤값으로 변수 1개를 만들고, 변수의 내용을 출력하시오.

import tensorflow as tf
# tf.set_random_seed(777) 

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

print(W)

W = tf.Variable([0.3], tf.float32)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# aaa = sess.run(W)
# print(aaa)
# sess.close()

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# aaa = W.eval()
# print(aaa)
# sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = W.eval(session=sess)
print(aaa)
sess.close()