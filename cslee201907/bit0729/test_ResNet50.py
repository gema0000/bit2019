# from tensorflow.python.keras.applications.vgg16 import VGG16

from keras.applications import ResNet50
# conv_base = Xception(weights='imagenet', include_top=False,
#                      input_shape=(150,150,3))  

conv_base = ResNet50() # (224, 224, 3) 
conv_base.summary()

# conv_base.summary()

from keras import models, layers

model = models.Sequential()
model.add(conv_base)
# model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

