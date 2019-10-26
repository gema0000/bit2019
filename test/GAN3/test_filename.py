from keras.layers import Dense
from keras.models import Sequential
import numpy as np

x = np.array([1,2,3])
y = np.array([1,2,3])

model = Sequential()
model.add(Dense(5, input_shape=(1,)))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))

# model.summary()

# model.trainable = False   # compile 위에다 써야한다.
model.compile(loss='mse', optimizer='adam')
# model.trainable = False

model.summary()

model.fit(x, y, epochs=5, batch_size=1)

import datetime
first_name = 'test'
second_name = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
filename = '_'.join([first_name, second_name])
print(filename)
model.save(filename+'.h5')

# print("===========================")
# print(model.layers[2].get_weights())
# print("===========================")
# print(model.layers[3].get_weights())
# # print("===========================")
# # print(model.layers[2].get_weights())

# print("===========================")
# print(model.layers[2].trainable_weights)
# print("===========================")
# print(model.layers[3].trainable_weights)

