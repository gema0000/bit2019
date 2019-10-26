from keras.models import Model
from keras.layers import Input, Dense, Reshape, BatchNormalization, UpSampling2D
from keras.layers import Conv2D, Activation, LeakyReLU, Dropout
from keras.layers import ZeroPadding2D, Flatten

# noise_shape=(100, )

# input = Input(noise_shape)
# x = Dense(128 * 7 * 7, activation="relu")(input)    # 6272
# x = Reshape((7, 7, 128))(x)
# x = BatchNormalization(momentum=0.8)(x)
# x = UpSampling2D()(x)
# x = Conv2D(128, kernel_size=3, padding="same")(x)
# x = Activation("relu")(x)
# x = BatchNormalization(momentum=0.8)(x)
# x = UpSampling2D()(x)
# x = Conv2D(64, kernel_size=3, padding="same")(x)
# x = Activation("relu")(x)
# x = BatchNormalization(momentum=0.8)(x)
# x = Conv2D(1, kernel_size=3, padding="same")(x)
# out = Activation("tanh")(x)
# model = Model(input, out)
# print("-- Generator -- ")
# model.summary()

img_shape=(28,28,1)

input = Input(img_shape)
x = Conv2D(32, kernel_size=3, strides=2, padding="same")(input)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
x = (LeakyReLU(alpha=0.2))(x)
x = Dropout(0.25)(x)
x = BatchNormalization(momentum=0.8)(x)
x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = BatchNormalization(momentum=0.8)(x)
x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(input, out)
print("-- Discriminator -- ")
model.summary()

from keras.datasets import mnist
import numpy as np
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)
return X_train

print(X_train)