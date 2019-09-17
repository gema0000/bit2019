#1. 데이터
import numpy as np

# x = np.array(range(1,101))
# y = np.array(range(1,101))
x = np.array([range(1000), range(3110,4110), range(1000)])
y = np.array([range(5010,6010)])

print(x.shape)
print(y.shape)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.4
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5
)

print(x_test.shape)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
model = Sequential()

from keras import regularizers
# model.add(Dense(5, input_dim = 3, activation ='relu'))
model.add(Dense(1000, input_shape = (3, ), activation ='relu',
                kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(1000, kernel_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(1000))
model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

# model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x, y, epochs=100, batch_size=3)
model.fit(x_train, y_train, epochs=30, batch_size=8,
          validation_data=(x_val, y_val))

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print("acc : ", acc)

y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

print("loss : ", loss) # 0.001 이하로 만들것.
