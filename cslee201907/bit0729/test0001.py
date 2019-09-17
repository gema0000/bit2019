from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
print(train_datagen)

