from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)
print(boston.keys())
print(boston.target)
print(boston.target.shape)

x = boston.data
y = boston.target

print(type(boston)) # <class 'sklearn.utils.Bunch'>

from sklearn.linear_model import LinearRegression #, Ridge, Lasso, 
# 모델 완성하시오.

# Student Test Version

# 학습하기 (1)
model = LinearRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

# 평가하기
y_pred= model.predict(x_test)
print("LinearRegression score : ", score)
# print("Lasso score : ", score)

# 학습하기 (2)
model = Ridge()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

# 평가하기 (2)
y_pred= model.predict(x_test)
print("Ridge score : ", score)

# 학습하기 (3)
model = Lasso()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

# 평가하기 (3)
y_pred= model.predict(x_test)
print("Lasso score : ", score)