from keras.layers import Input, Dense, concatenate, Concatenate
from keras.models import Model, Sequential
# from keras.layers.merge import Concatenate
import numpy as np

x = np.array([1,2,3])
y = np.array([1,2,3])

first = Sequential()
first.add(Dense(1, input_shape=(1,), activation='sigmoid'))

second = Sequential()
second.add(Dense(1, input_shape=(1,), activation='sigmoid'))

# result = Sequential()
merged = Concatenate([first, second])
# ada_grad = Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
# result.add(merged)
# result.compile(optimizer=ada_grad, loss=_loss_tensor, metrics=['accuracy'])

# merged.summary()

merged.add(Dense(2, activation="relu"))
merged.add(Dense(1, activation="linear"))
merged.summary()

# result.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# result.fit([x, x],[y, y], epochs=100, batch_size=1)

# loss, acc = result.evaluate([x, x],[y, y], batch_size=1)
# print("acc : ", acc)

