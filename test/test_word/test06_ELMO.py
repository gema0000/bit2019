# https://wikidocs.net/33930

import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K

sees = tf.Session()
K.set_session(sess)
# 세션 초기화. 이는 텐서플로우 개념

elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
# 텐서플로 허브로부터 ELMO를 다운로드

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
