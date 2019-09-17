from keras.models import Model

model = Model(inputs=inputs, outputs=output)
model = multi_gpu_model(model, num_gpu)

def build_network(num_gpu=1, input_shape=None):
    inputs = Input(shape=input_shape, name = "input")

    # 합성곱 블록1
    conv1 = Conv2D (4, kernel_size=(3,3), activation='relu', name='conv_1')(inputs)
    batch1 = BatchNormalization(name='batch_norm_1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2), name='pool_1')(batch1)

    # 합성곱 블록2
    conv2 = Conv2D (32, kernel_size=(3,3), activation='relu', name='conv_2')(pool1)
    batch2 = BatchNormalization(name='batch_norm_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2), name='pool_2')(batch2)

    # 완전 연결 계층들
    flatten = Flatten()(pool2)
    fc1 = Dense(512, activation='relu', name='fc1')(flatten)
    d1 = Dropout(rate=0.2, name='dropout1')(fc1)
    fc2 = Dense(256, activation='relu', name='fc2')(d1)
    d2 = Dropout(rate=0.2, name='dropout2')(fc2)

    # 출력 계층
    output = Dense(10, activation='softmax', name='softmax')(d2)

    model = Model(inputs=inputs, outputs=output)
    if num_gpu > 1:
        model = multi_gpu_model(model, num_gpu)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model                  

model =build_network(num_gpu=1, input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

model.fit(x=X_train, y=Y_train,
          batch_size = 32,
          epochs = 200,
          validation_data=(val_x, val_y), 
          verbose=1, callbacks=callbacks )





