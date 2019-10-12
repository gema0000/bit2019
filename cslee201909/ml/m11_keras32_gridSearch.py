# keras32_hyperParameter.py -> gridSearchCV 로 변경

# Student Test Version

#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy 
import os 
import tensorflow as tf

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# import matplotlib.pyplot as plt 
# digit = X_train[44] # 6만가지의 이미지 중 원하는 n번째 이미지를 보여줘라.
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show() # jupyter notebook이나 구글 colab 환경에서는 plt.show() 코드를 쓰지 않아도 이미지가 출력된다.

# X_train = X_train.reshape(X_train.shape[0], 28 ,28 ,1).astype('float32') / 255 # 6만행(무시) 나머지는 아래 input_shape값이 된다.
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255 # 0~1 사이로 수렴(minmax)시키기 위해 minmaxscaler같은거 필요없이 각 픽셀당 255의 값을 나누어서 데이터 전처리를 하는 과정
X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype('float32') / 255 # 밑에서 나올 784(28*28)로 바꿔주기
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float32') / 255 # CNN용을 DNN용으로 와꾸 바꿔준것
Y_train = np_utils.to_categorical(Y_train) # One Hot Incoding으로 데이터를 변환시킨다. 분류
Y_test = np_utils.to_categorical(Y_test)

# 하이퍼 파라미터 최적화
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(784, ), name='input') # 784 = 28*28 => mnist.....
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name='output')(x) 
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_hyperparameters(): # 45개의 옵션(batches*optimizers*dropout)을 랜덤으로 돌린다.
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta'] # 용도에 맞게 쓰자.
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함. (mnist에서 쓸듯)
# from keras.wrappers.scikit_learn import KerasRegressor # 사이킷런의 교차검증을 keras에서 사용하기 위해 wrapping함
model = KerasClassifier(build_fn=build_network, verbose=1) # verbose=0 위에서 만든 함수형 모델 당겨옴.

hyperparameters = create_hyperparameters() # batch_size, optimizer, dropout의 값을 반환해주는 함수 wrapping해옴.

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 그리드 서치
kfold_cv = KFold(n_splits=5, shuffle=True)
clf = GridSearchCV( estimator=model, param_grid=hyperparameters, cv=kfold_cv)
clf.fit(X_train, Y_train)
print("최적의 매개 변수 = ", clf.best_estimator_)

# 최적의 매개 변수로 평가하기
y_pred = clf.predict(X_test)
print("최종 정답률 = ", accuracy_score(Y_test, y_pred))
last_score = clf.score(X_test, Y_test)
print("최종 정답률 = ", last_score)

'''
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model,
                             param_distributions=hyperparameters,
                             n_iter=10, n_jobs=1, cv=3, verbose=1)
                             # 작업이 10회 수행, 3겹 교차검증 사용(3조각을 나눠서 검증). n_jobs는 알아서 찾아볼 것.
# KFold가 5번 돌았다면 얘는 랜덤하게 돈다. 이 작업을 하는 것은 위의 하이퍼파라미터 중 최적의 결과를 내는 파라미터들을 찾기 위함.
# search.fit(data["X_train"], data["Y_train"])
search.fit(X_train, Y_train) # 데이터 집어넣기!
print(search.best_params_)
'''