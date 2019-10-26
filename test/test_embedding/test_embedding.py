from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Reshape, Conv1D
from keras.layers import MaxPool1D, Flatten
from keras.models import Sequential
# crf = CRF(7)

'''
model = Sequential()
model.add(Embedding( 30, 100, input_length = 20))
model.add(Bidirectional(LSTM(units=32, return_sequences=True))) # recurrent_dropout=0.2
model.add(BatchNormalization())
# model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
# model.add(BatchNormalization())
# model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
# model.add(BatchNormalization())
model.add(TimeDistributed(Dense(100, activation="relu"))) # imeDistributed wrapper를 사용하여 3차원 텐서 입력을 받을 수 있게 확장해 주어야 한다.
# model.add(crf)
'''

# 안되는 거
# # from keras_contrib.layers import CRF
# from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, BatchNormalization, Conv2D, MaxPool2D, Reshape, Conv1D, MaxPool1D, Flatten
# crf = CRF(7)
# model = Sequential()
# model.add(Embedding(len(word_index), 100, input_length = 20))
# model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout = 0.2)))
# # model.add(Reshape(sh))
# model.add(Conv2D(128, (2,2), padding = 'same'))
# model.add(MaxPool2D(2,2))
# model.add(Conv2D(128, (2,2), padding = 'same'))
# model.add(MaxPool2D(2,2))
# model.add(BatchNormalization())
# model.add(TimeDistributed(Dense(100, activation="relu")))
# model.add(Reshape((250, 100)))
# model.add(crf)


model = Sequential()
model.add(Embedding(30, 100, input_length = 20))
model.add(Bidirectional(LSTM(units=100 * 5, recurrent_dropout = 0.2)))
# model.add(LSTM(units=100 * 5, recurrent_dropout = 0.2))
model.add(Reshape([100, 10]))
model.add(Conv1D(100, 2, padding = 'same'))
model.add(MaxPool1D(2))
model.add(Conv1D(100, 2, padding = 'same'))
model.add(MaxPool1D(2))
model.add(BatchNormalization())

# model.add(TimeDistributed(Dense(100, activation="relu")))
model.add(Dense(100, activation="relu"))
model.add(Reshape((25, 100, 1)))

model.summary()