from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])
print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape((x.shape[0], x.shape[1], 1))
print("x.shape : ", x.shape)

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(Dense(5))
model.add(Dense(1))

# model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100)

x_input = array([6,7,8]) # 1,3, ?????
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)

'''
과제
1,2,3 / 2,3,4 .... 10,11,12 로 데이터를 변경 (10,4)
y 는 4~13
새로운 값은 11,12,13 으로 14를 예측해낼것
70 80 90
'''


