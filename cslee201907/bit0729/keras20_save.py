
#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
model = Sequential()

from keras import regularizers
# model.add(Dense(5, input_dim = 3, activation ='relu'))
model.add(Dense(100, input_shape = (3, ), activation ='relu',
                kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(10))
# model.add(BatchNormalization())
model.add(Dense(20))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(10))

# model.summary()

model.save('savetest01.h5')
print("저장 잘 됬다.")
