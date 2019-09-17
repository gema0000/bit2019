import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris.csv", encoding='utf-8')
print(iris_data)
print(iris_data.shape)
print(type(iris_data))

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# y2 = iris_data.iloc[:, 4]
# x2 = iris_data.iloc[:, 0:4]

print("======================")
print(x.shape)
print(y.shape)

# print(x2.shape)
# print(y2.shape)

#### sklearn에서는 분류 y값이 문자형이어도 가능하나, keras에서는 바꿔줘야함.
print(y)
print("====================")
from sklearn.preprocessing import LabelEncoder
e = LabelEncoder()
e.fit(y)
y1 = e.transform(y)
print(y1)
print(y.shape)
print(y1.shape)

from keras.utils import np_utils
# one-hot-encoding
y_encoded = np_utils.to_categorical(y1)


# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x,y_encoded, test_size=0.2, train_size=0.8, shuffle=True)

# print(y_test)     #str형식으로 저장되어 있다.   

from keras.models import Model, Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(16, activation="relu", input_shape=(4,)))
# model.add(Dense(5))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=4, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

print(y_predict)
print("정확도 : ", acc)


# print(x_train.shape)
# print(type(x_train))

# #  학습하기
# clf = SVC()
# clf.fit(x_train, y_train)

# # 평가하기
# y_pred = clf.predict(x_test)
# print("정답률 : ", accuracy_score(y_test, y_pred))


# from sklearn import datasets, svm
# # 데이터 읽어들이기
# iris = datasets.load_iris()
# print("target = ", iris.target) # 레이블 데이터
# print("data = ", iris.data)     # 관측 데이터
# print(type(iris.data))          # numpy.ndarray
