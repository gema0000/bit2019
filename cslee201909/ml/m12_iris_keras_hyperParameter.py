# m05_iris_keras.py 를 RandomSearch 적용

# Student Test Version

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC   # 분류모델
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./_data/csv/iris.csv", encoding='utf-8',
                        names=['a', 'b', 'c', 'd', 'y'])
# print(iris_data)
# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "y"] # .loc 레이블로 나누기(레이블은 실질적인 데이터가 아님)
x = iris_data.loc[:,["a", "b", "c", "d"]]

# print(y)
# print(y.shape)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y) # 0 = setosa, 1 = versicolor, 2 = virginica
# print(pd.value_counts(y_encoded))
# print(y_encoded)

# 범주형으로 전환
y_encoded = np_utils.to_categorical(y_encoded, 3)
# print(y_encoded)
# print(y_encoded.shape)

x_array = np.array(x)
# print(x_array)
# print(x_array.shape) # (150,4)
# print("==============================")
# print(x.shape) # (150, 4)
# print(y.shape) # (150,)
# print(x2.shape)
# print(y2.shape)

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x_array,y_encoded, test_size=0.2, train_size=0.8, shuffle=True)

# print(x_train.shape) # (120, 4)
# print(x_test.shape) # (30, 4)
# print(y_train.shape) # (120, 3)
# print(y_test.shape) # (30, 3)


# 하이퍼 파라미터 최적화
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import numpy as np

def build_network(optimizer='adam'):
    model = Sequential()
    model.add(Dense(32, input_dim=4, activation='relu'))
    model.add(Dense(16)) # 여기서부턴 DNN연산모델
    model.add(Dense(3, activation = 'softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_hyperparameters(): # 45개의 옵션(batches*optimizers*dropout)을 랜덤으로 돌린다.
    batches = [10,20,30,40,50,100,120]
    optimizers = ['rmsprop', 'adam', 'adadelta'] # 용도에 맞게 쓰자.
    dropout = np.linspace(0.1, 0.5, 5)
    epochs = [100, 200, 300, 400, 500]
    return{"batch_size":batches, "optimizer":optimizers, "epochs":epochs} #, "keep_prob":dropout}

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함. (mnist에서 쓸듯)
# from keras.wrappers.scikit_learn import KerasRegressor # 사이킷런의 교차검증을 keras에서 사용하기 위해 wrapping함
model = KerasClassifier(build_fn=build_network, verbose=1) # verbose=0 위에서 만든 함수형 모델 당겨옴.

hyperparameters = create_hyperparameters() # batch_size, optimizer, dropout의 값을 반환해주는 함수 wrapping해옴.

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model,
                             param_distributions=hyperparameters,
                             n_iter=10, n_jobs=1, cv=3, verbose=1)
                             # 작업이 10회 수행, 3겹 교차검증 사용(3조각을 나눠서 검증). n_jobs는 알아서 찾아볼 것.
# KFold가 5번 돌았다면 얘는 랜덤하게 돈다. 이 작업을 하는 것은 위의 하이퍼파라미터 중 최적의 결과를 내는 파라미터들을 찾기 위함.
# search.fit(data["x_train"], data["y_train"])
search.fit(x_train, y_train) # 데이터 집어넣기!

print(search.best_params_)

