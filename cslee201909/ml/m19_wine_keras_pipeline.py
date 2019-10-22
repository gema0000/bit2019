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
# print(x_scaled[:100])


# 학습 데이터와 평가 데이터로 분리하기 
x_train, x_test, y_train, y_test = train_test_split(
                                    x_scaled, y_encoded, test_size=0.2)

x_val, x_test, y_val, y_test = train_test_split( # train 60 val 20 test 20 으로 분할
                                                    x_test, y_test, random_state=66, test_size=0.5)                                    

# print(x_train.shape) #(3918, 11)
# print(x_test.shape) # (980, 11)
# print(y_train.shape) # (3918,)
# print(y_test.shape) # (980,)

# 하이퍼 파라미터 최적화
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import numpy as np

# 모델의 설정
def build_network(optimizer='adam'):
    model = Sequential()
    model.add(Dense(50, input_dim=11, activation='relu'))
    model.add(Dense(10))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_hyperparameters(): # 45개의 옵션(batches*optimizers*dropout)을 랜덤으로 돌린다.
    batches = [50,100,120]
    optimizers = ['rmsprop', 'adam', 'adadelta'] # 용도에 맞게 쓰자.
    dropout = np.linspace(0.1, 0.25, 0.5, 5)
    epochs = [1,2]
    return{"kerasclassifier__batch_size":batches, "kerasclassifier__optimizer":optimizers, "kerasclassifier__epochs":epochs} #, "keep_prob":dropout}
# 밑에서 make를 쓸때는 각 parameter앞에 kerasclassifier__를, 그냥 Pipeline 쓸때는 svc__를 써주면 됨!
from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함. (mnist에서 쓸듯)
# from keras.wrappers.scikit_learn import KerasRegressor # 사이킷런의 교차검증을 keras에서 사용하기 위해 wrapping함
model = KerasClassifier(build_fn=build_network, verbose=1) # verbose=0 위에서 만든 함수형 모델 당겨옴.

from sklearn.preprocessing import MinMaxScaler                                                   
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

hyperparameters = create_hyperparameters()

pipe = make_pipeline(MinMaxScaler(), model)

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=pipe,
                             param_distributions=hyperparameters,
                             n_iter=10, n_jobs=1, cv=3, verbose=1)
#                              # 작업이 10회 수행, 3겹 교차검증 사용(3조각을 나눠서 검증). n_jobs는 알아서 찾아볼 것.
# KFold가 5번 돌았다면 얘는 랜덤하게 돈다. 이 작업을 하는 것은 위의 하이퍼파라미터 중 최적의 결과를 내는 파라미터들을 찾기 위함.
# search.fit(data["x_train"], data["y_train"])

search.fit(x_train, y_train) # 데이터 집어넣기!

print(search.best_score_)
print(search.best_params_)