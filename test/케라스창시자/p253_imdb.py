# object arrays cannot be loaded when allow_pickle=False 에러가 뜬다.
# 넘파이 버전을 1.16.4 -> 1.16.1로 바꾸면 된다.  !pip install numpy==1.16.1

from keras.datasets import imdb
from keras import preprocessing

max_features = 10000
max_len = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
print(x_train.shape)
print(x_test.shape)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
print(x_train.shape)
print(x_test.shape)

from keras.layers import Embedding
embedding_layer = Embedding(1000, 64)


