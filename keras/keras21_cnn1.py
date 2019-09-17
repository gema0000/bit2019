from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D
model = Sequential()
model.add(Conv2D(7, (2,2), padding='same', #same, valid 디폴트
                 input_shape =(10,10,1)))
# model.add(Conv2D(16,(2,2)))
model.add(MaxPooling2D(3,3))
# model.add(Conv2D(8,(2,2)))
model.add(Flatten())

model.summary()

