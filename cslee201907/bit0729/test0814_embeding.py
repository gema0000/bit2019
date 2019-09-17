from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)

print(x_train.shape)    #(25000, )
print(y_train.shape)    #(25000, )

print(x_test.shape)    #(25000, )
print(y_test.shape)    #(25000, )





