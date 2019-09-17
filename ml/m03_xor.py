from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0,1,1,0]

# 2. 모델
# model = LinearSVC()
model = SVC()

# 3. 실행
model.fit(x_data, y_data)

# 4. 평가 예측
x_test = [[0,0], [1,0], [0,1], [1,1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측결과 : ", y_predict)
print("acc = ", accuracy_score([0,1,1,0], y_predict))
