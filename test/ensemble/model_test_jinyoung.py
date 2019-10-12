import numpy as np

from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D
from keras.layers.merge import concatenate
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

image_data = np.load('./test/data/npy/image_data.npy')
mask_data = np.load('./test/data/npy/mask_data.npy')
class_data = np.load('./test/data/npy/class_name.npy')
# print(image_data.shape)   # (263, 720, 1280)
# print(mask_data.shape)   # (263, 720, 1280)
# print(class_data.shape)   # (263,)

image_data = image_data.reshape(-1, 720,1280,1).astype('float32') /255
mask_data = mask_data.reshape(-1, 720,1280,1).astype('float32') /255

le = LabelEncoder()
le.fit(class_data)
class_data = le.transform(class_data)

class_data = np_utils.to_categorical(class_data,2)
# print(class_data.shape)


x1, xt1, y1, yt1 = train_test_split(
    image_data, mask_data, test_size = 0.2
)
xt1, xv1 , yt1, yv1 = train_test_split(
    xt1, yt1, test_size=0.5
)
# print(x1.shape, y1.shape)   # (210, 720, 1280, 1) (210, 720, 1280, 1)
# print(xt1.shape, yt1.shape)   # (26, 720, 1280, 1) (26, 720, 1280, 1)
# print(xv1.shape, yv1.shape)   # (27, 720, 1280, 1) (27, 720, 1280, 1)
y1 = y1.reshape((-1, 921600))

x2, xt2, y2, yt2 = train_test_split(
    image_data, class_data, test_size = 0.2
)
xt2, xv2, yt2, yv2 = train_test_split(
    xt2, yt2 , test_size=0.5
)
# print(x2.shape,y2.shape)   # (210, 720, 1280, 1) (210, 2)
# print(xt2.shape,yt2.shape)   # (26, 720, 1280, 1) (26, 2)
# print(xv2.shape, yv2.shape)   # (27, 720, 1280, 1) (27, 2)

x3, xt3, y3, yt3 = train_test_split(
    mask_data, class_data, test_size = 0.2
)
xt3, xv3, yt3, yv3 = train_test_split(
    xt3, yt3, test_size = 0.5
)
# print(x3.shape,y3.shape)   # (210, 720, 1280, 1) (210, 2)
# print(xt3.shape, yt3.shape)   # (26, 720, 1280, 1) (26, 2)
# print(xv3.shape, yv3.shape)   # (27, 720, 1280, 1) (27, 2)


# model1 
input1 = Input(shape=(720,1280,1))
conv1 = Conv2D(192, activation='relu', kernel_size=(4,4),padding='same')(input1)
conv1 = MaxPool2D(pool_size=2)(conv1)

# model2
input2 = Input(shape=(720,1280,1))
conv2 = Conv2D(192, activation='relu', kernel_size=(4,4),padding='same')(input2)
conv2 = MaxPool2D(pool_size=2)(conv2)

# model3
input3 = Input(shape=(720,1280,1))
conv3 = Conv2D(192, activation='relu',kernel_size=(4,4),padding='same')(input3)
conv3 = MaxPool2D(pool_size=2)(conv3)


# main model
merge1 = concatenate([conv1, conv2, conv3])
mm = Conv2D(256, activation='relu',kernel_size =(4,4),padding='same')(merge1)
mm = MaxPool2D(pool_size=2)(mm)
mm = Conv2D(512, activation='relu', kernel_size=(4,4),padding='same')(mm)
mm = MaxPool2D(pool_size=2)(mm)
mm = Conv2D(1024, activation='relu', kernel_size=(4,4), padding='same')(mm)
mm = MaxPool2D(pool_size=2)(mm)
mm = Conv2D(1024, activation='relu', kernel_size=(4,4), padding='same')(mm)
output = Flatten()(mm)

# output1
output1 = Dense(50,activation='relu')(output)
output1 = Dense(921600, activation='relu')(output1) # 921600

# output2 
output2 = Dense(50, activation='relu')(output)
output2 = Dense(2,activation='softmax')(output2)


model = Model(input=[input1, input2, input3], output=[output1, output2])

model.summary()

model.compile(loss=['mse', 'categorical_crossentropy'], optimizer='adam') #, metrics=['acc'])

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])


model.fit(
    [x1,x2,x3], [y1,y3], epochs=100    #, validation_data=([xv1,xv2,xv3],[yv1,yv2,yv3])
)

