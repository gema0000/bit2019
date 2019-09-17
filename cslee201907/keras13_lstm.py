from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
#1. 데이터 
X = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = array([4, 5, 6, 7])

print("X.shape : ", X.shape)
print("y.shape : ", y.shape)

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X.shape : ", X.shape)
print("y.shape : ", y.shape)


#2. 모델 구성
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#3. 실행
model.fit(X, y, epochs=1000, verbose=2)
# demonstrate prediction
x_input = array([6, 7, 8])       # 70, 80, 90 => ?
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)
'''