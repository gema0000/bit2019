import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#1. 데이터
a = np.array(range(1,101))
batch_size = 2
size =5
def split_5(seq, size):
    aaa = []
    for i in range(len(a)- size + 1):
        subset = a[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
print("=================")
print(dataset)
print(dataset.shape)

x_train = dataset[:,0:4]
y_train = dataset[:, 4]

x_train = np.reshape(x_train, (len(x_train), size-1, 1))

x_test = x_train + 100
y_test = y_train + 100

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_test[0])

#2. 모델구성
model = Sequential()
model.add(LSTM(128, batch_input_shape=(batch_size,4,1),
               stateful=True))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

num_epochs = 100

for epoch_idx in range(num_epochs):
    print('epochs : ' + str(epoch_idx))
    model.fit(x_train, y_train, epochs=1, batch_size=batch_size,
              verbose=2, shuffle=False,
              validation_data=(x_test, y_test))
    model.reset_states()

mse, _ = model.evaluate(x_train, y_train, batch_size=batch_size)
print("mse : ", mse)
model.reset_states()

y_predict = model.predict(x_test, batch_size=batch_size)

print(y_test[0:5])
print(y_predict[0:5])


# ################ 실습 ##############################
# 1. mse값을  1 이하로 만들것
# 	-> 3개이상 히든레이어 추가할것. 
#                -> 드랍아웃 또는 batchnormalization적용
# 2. RMSE함수 적용
# 3. R2함수 적용

# 4. earlyStopping 기능 적용
# 5. tensorboard 적용

# 6. matplotlib 이미지 적용 mse/epochs
# #####################################################







