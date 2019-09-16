import numpy as np
from keras.layers import Input, Dense, Conv2D, Lambda, concatenate, MaxPool2D, Reshape, Flatten, UpSampling2D
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras import backend as K
from keras import objectives
from keras.losses import mse
import matplotlib.pyplot as plt
import keras.losses

# load data
x = np.load("./data/cvaex.npy") # 정면을 포함한 각도별 10개의 이미지
y = np.load("./data/cvaey.npy") # 각 각도의 라벨링 0~9
test = np.load("./data/test.npy")
x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
x = x.astype("float32") / 255
print(x.shape)
print(y.shape)
x_train = x[:280]
y_train = y[:280]
print(x_train.shape)
print(y_train.shape)
x_test = x[280:]
y_test = y[280:]
print(x_test.shape)
print(y_test.shape)
print(y[:10])
# train은 28명의 사람의 각각의 10개의 각도 test는 2명의 사람

# one_hot_encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("y_train.shape = ", y_train.shape ,"y_test.shape = ",y_test.shape)

# parameters
batch_size = 1
epoch = 1
# hidden = 512
z_dim = 2
img_size = x_train.shape[2]

# encoder
inputs = Input(shape = (img_size, img_size, 1), name = 'encoder_input') # 28 * 28,
condition = Input(shape = (10,), name= 'labels')
layer_c = Dense(img_size * img_size)(condition)
layer_c = Reshape((img_size, img_size, 1))(layer_c)

x = concatenate([inputs, layer_c])
x_encoded = Conv2D(32, 3, padding='same', activation='relu')(x)
x_encoded = MaxPool2D(2,2)(x_encoded)
x_encoded = Conv2D(16, 3, padding='same', activation='relu')(x_encoded)
x_encoded = MaxPool2D(2,2)(x_encoded)
x_encoded = Conv2D(8, 3, padding='same', activation='relu')(x_encoded)

z_shape = K.int_shape(x_encoded)

x_encoded = Flatten()(x_encoded)
x_encoded = Dense(img_size//2, activation="relu")(x_encoded)
z_mean = Dense(z_dim)(x_encoded)
z_log_val = Dense(z_dim)(x_encoded)

def sampling(args):
    from keras import backend as K
    z_mean, z_log_val = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape = (batch, dim))
    return z_mean + K.exp(0.5 * z_log_val) * epsilon

z = Lambda(sampling, output_shape=(z_dim,))([z_mean, z_log_val])

encoder = Model([inputs, condition], [z_mean, z_log_val, z])
encoder.summary()
# encoder.compile(optimizer='adam', loss='mse')   # 추가

z_input = Input(shape=(z_dim,))
z_con = concatenate([z_input, condition])
layer_z = Dense(z_shape[1] * z_shape[2] * z_shape[3], activation="relu")(z_con)
layer_z = Reshape((z_shape[1], z_shape[2], z_shape[3]))(layer_z)

z_decoded = Conv2D(8, 3, padding='same', activation='relu')(layer_z)
z_decoded = Conv2D(16, 3, padding='same', activation='relu')(z_decoded)
z_decoded = UpSampling2D(2,)(z_decoded)
z_decoded = Conv2D(32, 3, padding='same', activation='relu')(z_decoded)
z_decoded = UpSampling2D(2,)(z_decoded)
outputs = Conv2D(1, 3, padding='same', activation='sigmoid')(z_decoded)

decoder = Model([z_input, condition], outputs)
decoder.summary()
# decoder.compile(optimizer='adam', loss='mse')   # 추가

outputs = decoder([encoder([inputs, condition])[2], condition])

# cvae = Model([inputs, condition], outputs)

# loss
# reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
# reconstruction_loss *= img_size * img_size
# kl_loss = 1 + z_log_val - K.square(z_mean) - K.exp(z_log_val)
# kl_loss = K.sum(kl_loss, axis=-1)
# kl_loss *= -0.5 * 1.0

# cvae_loss = K.mean(reconstruction_loss + kl_loss)
# keras.losses.cvae_loss = cvae_loss        # 중복 필요없음.


# build model
def aaa(outputs = outputs):
    return K.mean(K.square(outputs))
# import test0916 as te

cvae = Model([inputs, condition], outputs)

# cvae.add_loss(aaa(outputs))
cvae.add_loss(K.mean(K.square(outputs)))
# cvae.add_loss(cvae_loss)
cvae.compile(optimizer='adam')
# cvae.compile(optimizer='adam', loss='mse')
# cvae.compile(optimizer='adam', loss=keras.losses.cvae_loss)
cvae.summary()


# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []
#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))

# train
# history = LossHistory()
cvae.fit([x_train, y_train], shuffle=True, epochs=epoch, batch_size=batch_size, 
                             validation_data=([x_test, y_test], None), verbose=1) #, 
                            #  callbacks=[history])

# cvae.save("./data/cvae.h5")
encoder.save("./data/cvae_encoder.h5")
decoder.save("./data/cvae_decoder.h5")

# print(history.losses)

from keras.models import load_model
# K.mean(K.square(outputs))
# cvae = load_model("./data/cvae.h5" )
cvae2 = load_model("./data/cvae_encoder.h5" )
cvae2 = load_model("./data/cvae_decoder.h5" )

print('1개는 로드 된다.')
# cvae = load_model("./data/cvae.h5" )

# cvae = load_model("./data/cvae.h5", custom_objects={'loss': aaa  } )
'''

# import tensorflow as tf
# cvae = tf.keras.models.load_model("./data/cvae.h5")

# cvae = load_model("./data/cvae.h5", custom_objects={'loss': mse} )
'''
