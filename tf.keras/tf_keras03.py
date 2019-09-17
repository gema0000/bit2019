import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=2, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, 
                             scale=(0.5 + t*t/3), 
                             size=None)
        y.append(r)
    return  x, 1.726*x -0.84 + np.array(y)

x, y = make_random_data() 

# plt.plot(x, y, 'o')
# plt.show()

import tensorflow as tf

x_train, y_train = x[:150], y[:150]
x_test, y_test = x[150:], y[150:]

'''
model = tf.keras.Sequential()
# from keras.models import Sequential
# model = Sequential()

model.add(tf.keras.layers.Dense(units=1, input_dim=1))
'''
###################### 추가
input = tf.keras.Input(shape=(1,))  # 함수형모델
output = tf.keras.layers.Dense(1)(input)

model = tf.keras.Model(input, output)
###################################

model.summary()

model.compile(optimizer='sgd', loss='mse')
history = model.fit(x_train, y_train, epochs=300, 
                    validation_split=0.3)

epochs = np.arange(1, 300+1)
plt.plot(epochs, history.history['loss'], label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

