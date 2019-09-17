from keras.models import Sequential
from keras.layers import Conv2D

filter_size = 32
kernel_size = (3,3)
model = Sequential()
model.add(Conv2D(filter_size, kernel_size, #padding='valid', 
                 input_shape = (28,28,1)))
# from keras.layers import Activation
# model.add(Activation('relu'))

model.add(Conv2D(16, (3,3)))

from keras.layers import MaxPooling2D
pool_size = (2,2)
model.add(MaxPooling2D(pool_size))

model.summary()





