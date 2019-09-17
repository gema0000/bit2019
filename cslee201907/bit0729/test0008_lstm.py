import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,101))

size =8
def split_5(seq, size):
    aaa = []
    for i in range(len(a)- size + 1):
        subset = a[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
print("====================")
print(dataset)

x_train = dataset[:,0:4]
y_train = dataset[:,4:6]

print("====================")
print(x_train)  # 5,4
print(y_train)  # 5,2


print(x_train.shape)    # (5,4)        
print(y_train.shape)    # (5, )


# x_train = np.reshape(x_train, (6, 4, 1))
x_train = np.reshape(x_train, (len(a)-size+1, 2, 2))

print(x_train.shape)

'''
x_test = np.array([[[11],[12],[13],[14]], [[12],[13],[14],[15]], 
                  [[13],[14],[15],[16]], [[14],[15],[16],[17]]])
y_test = np.array([15, 16, 17, 18])

print(x_test.shape)
print(y_test.shape)

#2. 모델 구성
model = Sequential()

model.add(LSTM(32, input_shape=(4,1), return_sequences=True))
model.add(LSTM(10,return_sequences=True))
model.add(LSTM(10,return_sequences=True))
model.add(LSTM(10,return_sequences=True))
model.add(LSTM(10,return_sequences=True))
model.add(LSTM(10,return_sequences=True))
model.add(LSTM(10,return_sequences=True))
model.add(LSTM(10,return_sequences=True))
model.add(LSTM(10,return_sequences=True))
model.add(LSTM(10))

model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(loss ='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
          callbacks=[early_stopping]  )

loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)
    
'''




