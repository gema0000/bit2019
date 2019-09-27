from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt("./_data/csv/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

# 모델의 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# 모델 실행
model.fit(X, Y, epochs=200, batch_size=10)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))

# Accuracy: 0.7852

# 실습1. keras 모델을 ML모델로 변경
# 실습2. LinearSVC 의 옵션을 바꿔가면서 acc 확인.
# 실습3. ml -> keras 변경

