#1. 데이터
import numpy as np

xxx1 = np.array([range(100), range(311,411)])
xxx2 = np.array([range(101, 201), range(311,411)])

yyy1 = np.array([range(501,601), range(111,11,-1)])
yyy2 = np.array([range(401,501), range(211,311)])
yyy3 = np.array([range(401,501), range(211,311)])


print(xxx1.shape)
xxx1 = np.transpose(xxx1)
xxx2 = np.transpose(xxx2)
yyy1 = np.transpose(yyy1)
yyy2 = np.transpose(yyy2)
yyy3 = np.transpose(yyy3)

print(xxx1.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    xxx1, yyy1 , random_state=66, test_size=0.4
)
x1_val, x1_test, y1_val, y1_test = train_test_split(
    x1_test, y1_test , random_state=66, test_size=0.5
)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    xxx2, yyy2 , random_state=66, test_size=0.4
)
x2_val, x2_test, y2_val, y2_test = train_test_split(
    x2_test, y2_test , random_state=66, test_size=0.5
)
y3_train, y3_test = train_test_split(
    yyy3 , random_state=66, test_size=0.4
)
y3_val, y3_test = train_test_split(
    y3_test , random_state=66, test_size=0.5
)

# print('x1_train : ', x1_train)
print('x1_train.shape : ', x1_train.shape)
print('x1_test.shape : ', x1_test.shape)
print('x1_val.shape : ', x1_val.shape)
print('x2_train.shape : ', x2_train.shape)
print('x2_test.shape : ', x2_test.shape)
print('x2_val.shape : ', x2_val.shape)
# print('x3_train.shape : ', x3_train.shape)
# print('x3_test.shape : ', x3_test.shape)
# print('x3_val.shape : ', x3_val.shape)
print('y1_train.shape : ', y1_train.shape)
print('y1_test.shape : ', y1_test.shape)
print('y1_val.shape : ', y1_val.shape)
print('y2_train.shape : ', y2_train.shape)
print('y2_test.shape : ', y2_test.shape)
print('y2_val.shape : ', y2_val.shape)
print('y3_train.shape : ', y3_train.shape)
print('y3_test.shape : ', y3_test.shape)
print('y3_val.shape : ', y3_val.shape)


#2. 모델구성
# from keras.models import Sequential
from keras.layers import Dense, Input #,Concatenate
from keras.models import Sequential, Model
from keras.layers.merge import concatenate
# model = Sequential()

# model 1
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu')(input1)
# model 2
input2 = Input(shape=(2,))
dense2 = Dense(50, activation='relu')(input2)
dense21 = Dense(50, activation='relu')(dense2)
# merge
# merge = Concatenate()([dense1, dense2])
merge1 = concatenate([dense1, dense21])

########################################## 오전은 위까지

output_11 = Dense(10)(merge1)
output_12 = Dense(5)(output_11)
merge2 = Dense(3)(output_12)

###########################################
output_1 = Dense(30)(merge2)
output1 = Dense(2)(output_1)

output_2 = Dense(70)(merge2)
output2 = Dense(2)(output_2)

output_3 = Dense(70)(merge2)
output3 = Dense(2)(output_3)

model = Model(inputs=[input1, input2], 
              outputs=[output1, output2, output3])

model.summary()


#3. 훈련
model.compile(loss=['mse','mae', 'mse'], optimizer='adam') #, metrics=['mse'])
model.fit([x1_train, x2_train],[y1_train, y2_train, y3_train], epochs=100, batch_size=1) #,
        #   validation_data=([x1_val, x2_val],[y1_val, y2_val, y3_val]))

#4. 평가 예측
# loss, acc = model.evaluate([x1_test, x2_test],[y1_test, y2_test], batch_size=1)
# acc = model.evaluate([x1_test, x2_test],[y1_test, y2_test, y3_test], batch_size=1)
aaa = model.evaluate([x1_test, x2_test],[y1_test, y2_test, y3_test], batch_size=1)

print("loss : ", aaa)

# y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])
# # y_predict = model.predict(x4)
# print(y1_predict, y2_predict)


'''
#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))    
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1) 
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2) 

# R2 구하기
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)
# print("R2 : ", r2_y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
print("R2_1 : ", r2_y1_predict)
print("R2_2 : ", r2_y2_predict)
print("R2 : ", (r2_y1_predict + r2_y2_predict)/2 )
'''
 