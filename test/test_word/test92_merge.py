from keras.layers import Dense, merge, Add
# from keras.layers import Dense, merge, concatenate, Concatenate
from keras.models import Model, Sequential
import numpy as np
from keras.layers.merge import concatenate, Concatenate


x = np.array([1,2,3])
y = np.array([1,2,3])

a = Sequential()
a.add(Dense(32, input_dim=1))

b = Sequential()
b.add(Dense(24, input_dim=1))

# merged = merge([a, b], mode = 'concat')
merged = Concatenate([a, b])

# middle1 = Dense(10)(merged)
# middle2 = Dense(1)(middle1)

# model = Model(inputs = [a, b], outputs = [middle2])
# model.summary()

model = Sequential()
from keras import layers
layers.Add(merged)
# model.add(Dense(10))
# model.add(Dense(1))

# # c.summary()

# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# model.fit([x, x], y, epochs=100, batch_size=1)

# loss, acc = result.evaluate([x, x],[y, y], batch_size=1)
# print("acc : ", acc)

