# https://www.kaggle.com/anandad/classify-fashion-mnist-with-vgg16

#1. 데이터 구성
from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    #(60000, 28, 28)
print(y_train.shape)    #(60000, )

# Convert the images into 3 channels
x_train=np.dstack([x_train] * 3)    
x_test=np.dstack([x_test]*3)
print(x_train.shape,x_test.shape)   #(60000, 28, 84)

from keras.utils import np_utils
x_train = x_train.reshape(x_train.shape[0], 28, 28, 3).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 3).astype('float32') / 255
print(x_train.shape)     #(60000, 28, 28, 1)
print(x_test.shape)      #(60000, )

# Resize the images 48*48 as required by VGG16
from keras.preprocessing.image import img_to_array, array_to_img
x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])
#train_x = preprocess_input(x)
print(x_train.shape, x_test.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)     #(60000, 10)
print(y_test.shape)      #(10000, 10)


#2-1. 모델 구성 1. 땡겨오기
# from tensorflow.python.keras.applications.vgg16 import VGG16
from keras.applications import VGG16
from keras.applications import MobileNetV2
# conv_base = VGG16(weights='imagenet', include_top=False,
#                      input_shape=(48,48,3))  # 32,32 이상이 되어야한다.
# include_top : 완전분류기에 연결

conv_base = VGG16() # (224, 224, 3) 
conv_base.summary()

#2-2. 모델 구성 2. 연결하기
from keras import models, layers
model = models.Sequential()
model.add(conv_base)
# model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

'''
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                    epochs=30, batch_size=200, verbose=1) # epochs=30
                    # callbacks=[early_stopping_callback]) #,checkpointer])

print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))

'''

# 실습
# 나머지 5개의 예약된 모델도 mnist로 완성하시오.
