#1. 데이터
import numpy as np

xxx = np.array([range(100), range(311,411)])
yyy = np.array([range(501,601), range(111,11,-1)])
# xxx = np.array([[1,2,3,4,5,6,7,8,9,10],
#                  [11,12,13,14,15,16,17,18,19,20]])
# yyy = np.array([[1,2,3,4,5,6,7,8,9,10],
#                  [11,12,13,14,15,16,17,18,19,20]])

print(xxx.shape)
xxx = np.transpose(xxx)
yyy = np.transpose(yyy)

print(xxx.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    xxx, yyy , random_state=66, test_size=0.4
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test , random_state=66, test_size=0.5
)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

model = Sequential()
model.add(Dense(200, input_dim=2, activation='relu', 
                kernel_regularizer=regularizers.l2(0.1)))
# model.add(Dense(200, input_dim=2, activation='relu')) 
model.add(Dense(3))
model.add(Dense(100))
model.add(Dense(4))
model.add(Dense(2))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=30, batch_size=1,
                    validation_data=(x_val, y_val))

print(hist.history.keys())

#4. 평가 예측

aaa = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", aaa)

# 시각화
import matplotlib.pyplot as plt
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# 손실 히스토리 요약
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['훈련', '테스트'], loc='upper left')
plt.show()


'''
# loss, acc = model.evaluate(x_test, y_test, batch_size=1)
# print("acc : ", acc)
# print("loss : ", loss)

y_predict = model.predict(x_test)
# y_predict = model.predict(x4)
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
'''




