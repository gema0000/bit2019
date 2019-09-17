# 실습 : acc도 99%로 올릴것

from numpy import array
from keras.layers import Dense, LSTM

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train.shape)    # (60000, 28, 28)
print(Y_train.shape)    # (60000, )
print(X_test.shape)
print(Y_test.shape)

# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train.shape)
print(Y_test.shape)


#2. 모델 구성
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(28, 28)))
# model.add(Dense(50))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. 실행
from keras.optimizers import Adam
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(X_train, Y_train, epochs=5, verbose=1, batch_size=128,
          callbacks=[early_stopping])

loss, acc = model.evaluate(X_test, Y_test, batch_size=128)

print("loss : ", loss)
print("acc : ", acc)
