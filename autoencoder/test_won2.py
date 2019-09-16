from keras.models import load_model
import numpy as np
from keras.layers import Input, Dense, Conv2D, Lambda, concatenate, MaxPool2D, Reshape, Flatten, UpSampling2D
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras import objectives
from keras.losses import mse
from keras import backend as K
import matplotlib.pyplot as plt


def aaa(outputs = outputs):
    return K.mean(K.square(outputs))

# K.mean(K.square(outputs))
# cvae = load_model("./data/cvae.h5" )

# import tensorflow as tf
# cvae = tf.keras.models.load_model("./data/cvae.h5")

cvae = load_model("./data/cvae.h5", custom_objects={'loss': mse} )

# cvae = load_model("./swh/model/cvae.h5")
# encoder = load_model("./data/cvae_encoder.h5")
# decoder = load_model("./data/cvae_decoder.h5")

'''
# load data
x = np.load("./npy/cvaex.npy") # 정면을 포함한 각도별 10개의 이미지
y = np.load("./npy/cvaey.npy") # 각 각도의 라벨링 0~9
test = np.load("./npy/test.npy")
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
y_label = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]) # 4 and 5 is front label


for i in range(len(x_test)):
    fig = plt.figure(figsize=(10,10))
    selected = x_test[i]
    selected = selected.reshape(28,28)
    plot = fig.add_subplot(1, 3, 1)
    plot.set_title("selected")
    plt.axis("off")
    plt.imshow(selected, cmap="Greys_r")
  
    real = x_test[(i // 10)*10]
    real = real.reshape(28,28)
    plot = fig.add_subplot(1, 3, 2)
    plot.set_title("real")
    plt.axis("off")
    plt.imshow(real, cmap="Greys_r")

    predictimg = cvae.predict([x_test[i].reshape(1,28,28,1), y_label])
    predictimg = predictimg.reshape(28,28)
    plot = fig.add_subplot(1, 3, 3)
    plot.set_title("predict")
    plt.axis("off")
    plt.imshow(predictimg, cmap="Greys_r")

    plt.show()
'''    
