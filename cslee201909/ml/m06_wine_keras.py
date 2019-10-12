# Student Test Version

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt 

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1) # "quality"라는 column만 drop하고 나머지는 x값이 된다.

# y 레이블 변경하기 --- (*2)
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

x = np.array(x)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# print(y_encoded[:100])

# 범주형으로 전환
y_encoded = np_utils.to_categorical(y_encoded, 3)

# 정규화
# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print(x_scaled[:100])


# 학습 데이터와 평가 데이터로 분리하기 
x_train, x_test, y_train, y_test = train_test_split(
                                    x_scaled, y_encoded, test_size=0.2)

x_val, x_test, y_val, y_test = train_test_split( # train 60 val 20 test 20 으로 분할
    x_test, y_test, random_state=66, test_size=0.5)

print(x_train.shape) #(3918, 11)
print(x_test.shape) # (980, 11)
print(y_train.shape) # (3918,)
print(y_test.shape) # (980,)


# 모델의 설정
model = Sequential()
model.add(Dense(50, input_dim=11, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.summary()

# 모델 최적화 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto')

# 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=200, 
         validation_data= (x_val, y_val), callbacks=[early_stopping] )

# 평가하기
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1)
y_predict = label_encoder.inverse_transform(y_predict) # int로 encoding했던 str을 다시 불러오는 것

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)
    

# acc = 0.9326