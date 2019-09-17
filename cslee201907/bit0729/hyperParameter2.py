#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# import matplotlib.pyplot as plt
# digit = X_train[5900]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float32') / 255

print(Y_train.shape)
print(Y_test.shape)
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train.shape)
print(Y_test.shape)

# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

##############################################################################

# 하이퍼 파라미터 최적화
# grid search -> random search
# RandomSearchCV, RandomizedSearchCV

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np 

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(784, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden2')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden3')(inputs)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

from keras.wrappers.scikit_learn import KerasClassifier   # 사이킷런과 호환하도록 함.
# from keras.wrappers.scikit_learn import KerasRegressor  # 사이킷런과 호환하도록 함.
model = KerasClassifier(build_fn=build_network, verbose=1)  # verbose=0

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model, 
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=3, verbose=1)  
                            # 작업이 10회 수행, 3겹 교차검증 사용.
# search.fit(data["X_train"], data["Y_train"])
search.fit(X_train, Y_train)

print(search.best_params_)

