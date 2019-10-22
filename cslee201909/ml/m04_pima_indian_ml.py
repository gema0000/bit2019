# Student Test Version

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# # seed 값 생성
# seed = 0
# numpy.random.seed(seed)
# tf.set_random_seed(seed)

# 데이터 로드 
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",") # (.)현재폴더, (..)상위폴더
X = dataset[:, 0:8] # 당뇨병 여부를 알아보기 위한 나이, 가족력 등의 지표 8개
Y = dataset[:,8] # 당뇨병이 있냐 없냐를 0, 1로 분류(마지막 9열)

# scaler 사용
scaler = MinMaxScaler() # 0.7759 / 0.7727 / 0.7857 / 0.7889 C=11 / 0.7922 gamma=0.16
# scaler = StandardScaler() # 0.7857 / 0.7435
scaler.fit(X)
X_scaled = scaler.transform(X)

# 훈련데이터와 평가데이터 나누기
from sklearn.model_selection import train_test_split # 사이킷런의 분할기능
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, random_state=66, test_size=0.4 # test를 40%로 train을 60%의 양으로 분할
)

# 모델의 설정
# model = SVC(kernel='sigmoid', C=10, gamma=0.16)
# model = SVC(kernel='poly', C=5, gamma=0.16, degree=6, coef0=6)
# model = LinearSVC(C=11, loss='hinge', max_iter=1000, multi_class='ovr') # 0.7759 / 0.7857
model = KNeighborsClassifier(n_neighbors=6) # 0.7272
# model = KNeighborsRegressor(n_neighbors=1, algorithm='brute')

# 모델 실행
model.fit(X_train, Y_train)

# # 평가 예측
y_predict = model.predict(X_test)

# print(X_test, "의 예측결과 : \n", y_predict)
print("acc = ", accuracy_score(Y_test, y_predict)) 
# accuracy_score(원래값, 비교값) 단순비교이므로 분류모델에서만 사용
# 회귀모델의 경우 원래값이 1, 비교값이 0.99999여도 다른값으로 나오기 때문에 acc가 안나온다.