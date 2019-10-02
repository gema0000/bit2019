import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./_data/csv/iris.csv", encoding='utf-8', 
                        names=['a', 'b', 'c', 'd', 'y']) #, header=None)
print(iris_data)
print(iris_data.shape)
print(type(iris_data))

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "y"]
x = iris_data.loc[:,["a", "b", "c", "d"]]

# y2 = iris_data.iloc[:, 4]
# x2 = iris_data.iloc[:, 0:4]

print("======================")
print(x.shape)  # (150, 4)
print(y.shape)  # (150, )

# print(x2.shape)
# print(y2.shape)


# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, train_size=0.7, shuffle=True)


# print(y_test)     #str형식으로 저장되어 있다.   

# print(x_train.shape)
# print(x_test.shape)

# print(type(x_train))

#  학습하기
clf = SVC()
clf.fit(x_train, y_train)

# 평가하기
y_pred = clf.predict(x_test)
print("정답률 : ", accuracy_score(y_test, y_pred))  # 0.933 ~ 1.0

'''
# from sklearn import datasets, svm
# # 데이터 읽어들이기
# iris = datasets.load_iris()
# print("target = ", iris.target) # 레이블 데이터
# print("data = ", iris.data)     # 관측 데이터
# print(type(iris.data))          # numpy.ndarray
'''

# 실습 : knn으로 바꾸시오.