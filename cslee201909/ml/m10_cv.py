from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression,SVC

iris = load_iris()
model = LogisticRegression()
# model = SVC()


# score = cross_val_score(model, iris.data, iris.target)  
# 디폴트는 3겹교차검증, sk-learn 0.22 부터 5겹교차검증
score = cross_val_score(model, iris.data, iris.target, cv=5)  

print("교차 검증 점수 : ", score)

# 현재 버전 0.20.3 -> 0.22 이상으로 바꿔야 아래는 실행 된다.   현재 최고버전은 0.21 이다.
# from sklearn.model_selection import cross_validate
# score2 = cross_validate(model, iris.data, iris.target, cv=5, return_train_score=True)

# print(iris.data)
# print(iris.target)
