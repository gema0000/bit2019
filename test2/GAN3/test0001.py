
from keras.models import Sequential

# from keras_contrib.layers import CRF
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, BatchNormalization, Conv2D, MaxPool2D, Reshape, Conv1D, MaxPool1D, Flatten
# crf = CRF(7)
'''
model = Sequential()
model.add(Embedding(30, 100, input_length = 20))
model.add(Bidirectional(LSTM(units=32, return_sequences=True))) # recurrent_dropout=0.2
model.add(BatchNormalization())
# model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
# model.add(BatchNormalization())
# model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
# model.add(BatchNormalization())
model.add(TimeDistributed(Dense(100, activation="relu"))) # imeDistributed wrapper를 사용하여 3차원 텐서 입력을 받을 수 있게 확장해 주어야 한다.
# model.add(crf)
'''

# from keras_contrib.layers import CRF
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, BatchNormalization, Conv2D, MaxPool2D, Reshape, Conv1D, MaxPool1D, Flatten
# crf = CRF(7)
model = Sequential()
model.add(Embedding(30, 100, input_length = 20))
# model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout = 0.2)))
model.add(Bidirectional(LSTM(units=10*3, recurrent_dropout = 0.2)))
# model.add(LSTM(units=100, recurrent_dropout = 0.2))
model.add(Reshape(10,3,2))
model.add(Conv1D(128, 2, strides=1, padding = 'same'))
# model.add(MaxPool2D(2,2))
# model.add(Conv2D(128, (2,2), padding = 'same'))
# model.add(MaxPool2D(2,2))
# model.add(BatchNormalization())
# model.add(TimeDistributed(Dense(100, activation="relu")))
# model.add(Reshape((250, 100)))
# model.add(crf)

model.summary()