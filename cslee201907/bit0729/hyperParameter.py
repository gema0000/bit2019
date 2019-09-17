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
model = KerasClassifier(build_fn=build_network, verbose=0)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model, 
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=3, verbose=1)  
                            # 작업이 10회 수행, 3겹 교차검증 사용.
search.fit(data["train_X"], data["train_y"])
print(search.best_params_)



print('끗')