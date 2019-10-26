from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(
    rescale=1/255.,         # 스케일변환
    rotation_range=90.,     # 데이터 확장 관련
    width_shift_range=1.,
    height_shift_range=.5,
    shear_range=.8,
    zoom_range=.5,
    horizontal_flip=True,
    vertical_flip=True
)

iters = gen.flow_from_directory(
    'img',
    target_size=(32, 32),
    class_mode='binary',
    batch_size=5,
    shuffle=True
)

x_train_batch, y_train_batch = next(iters)

print('shape of x_train_batch : ', x_train_batch.shape)
print('shape of y_train_batch : ', y_train_batch.shape)

