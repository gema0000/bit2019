# m13_cancer_keras_hyperParameter.py 에
# 파이프라인을 연결할것.

# Student Test Version

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

cancer = load_breast_cancer() # 분류

X = cancer.data
y = cancer.target

# print(y.value_counts())


# 정규화
# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X[:100])

# print(X.shape) # (569, 30)
# print(y.shape) # (569,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# print(X_train.shape) # (398,30)
# print(X_test.shape) # (171, 30)
# print(y_train.shape) # (398,)
# print(y_test.shape) # (171,)

# 하이퍼 파라미터 최적화
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import numpy as np

def build_network(optimizer='adam'):
    model = Sequential()
    model.add(Dense(32, input_dim=30, activation='relu'))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_hyperparameters(): # 45개의 옵션(batches*optimizers*dropout)을 랜덤으로 돌린다.
    batches = [50,100,120]
    optimizers = ['rmsprop', 'adam', 'adadelta'] # 용도에 맞게 쓰자.
    dropout = np.linspace(0.1, 0.25, 0.5, 5)
    epochs = [10, 50, 100]
    return{"kerasclassifier__batch_size":batches, "kerasclassifier__optimizer":optimizers, "kerasclassifier__epochs":epochs} #, "keep_prob":dropout}
# 밑에서 make를 쓸때는 각 parameter앞에 kerasclassifier__를, 그냥 Pipeline 쓸때는 svc__를 써주면 됨!
from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함. (mnist에서 쓸듯)
# from keras.wrappers.scikit_learn import KerasRegressor # 사이킷런의 교차검증을 keras에서 사용하기 위해 wrapping함
model = KerasClassifier(build_fn=build_network, verbose=1) # verbose=0 위에서 만든 함수형 모델 당겨옴.

from sklearn.preprocessing import MinMaxScaler                                                   
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

hyperparameters = create_hyperparameters() # batch_size, optimizer, dropout의 값을 반환해주는 함수 wrapping해옴.

# from sklearn.model_selection import RandomizedSearchCV
# search = RandomizedSearchCV(estimator=model,
#                              param_distributions=hyperparameters,
#                              n_iter=10, n_jobs=1, cv=3, verbose=1)

# pipe = Pipeline([("scaler", MinMaxScaler()), ('model', model)])
pipe = make_pipeline(MinMaxScaler(), model) # == pipe = Pipeline([("minmaxscaler", MinMaxScaler()), ('kerasclassifier', model)])

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=pipe,
                             param_distributions=hyperparameters,
                             n_iter=10, n_jobs=1, cv=3, verbose=1)
#                              # 작업이 10회 수행, 3겹 교차검증 사용(3조각을 나눠서 검증). n_jobs는 알아서 찾아볼 것.
# KFold가 5번 돌았다면 얘는 랜덤하게 돈다. 이 작업을 하는 것은 위의 하이퍼파라미터 중 최적의 결과를 내는 파라미터들을 찾기 위함.
# search.fit(data["x_train"], data["y_train"])

search.fit(X_train, y_train) # 데이터 집어넣기!

print(search.best_params_)