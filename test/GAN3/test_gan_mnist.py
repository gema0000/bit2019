from keras.layers import Dense, Reshape, BatchNormalization, Conv2DTranspose
from keras.layers import Activation, LeakyReLU, Conv2D, Input, Flatten
from keras.models import Model, load_model
from keras.datasets import mnist
import numpy as np
from keras.optimizers import RMSprop
import os
import matplotlib.pyplot as plt

def build_generator(inputs, image_size):
    image_resize = image_size // 4
    kernel_size = 5
    layers_filters = [128, 64, 32, 1]

    x = Dense(image_resize * image_resize * layers_filters[0])(inputs)
    x = Reshape((image_resize, image_resize, layers_filters[0]))(x)

    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)
    x = Activation('sigmoid')(x)
    generator = Model(inputs, x, name='generator')
    return generator                        

def build_discriminator(inputs):
    kernel_size = 5
    layers_filters = [32, 64, 128, 256]
    
    x = inputs
    for filters in layers_filters:
        if filters == layers_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(alpha=0.2)(x)        
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs, x, name='discriminator')
    return discriminator

def build_and_train_models():
    (x_train, _), (_, _) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255

    model_name = 'dcgan_mnist'

    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)

    # discriminator model 만들기
    inputs = Input(shape=input_shape, name='discriminator_input')
    discriminator = build_discriminator(inputs)

    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compilie(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
    discriminator.summary()

    # generator model 만들기
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = build_generator(inputs, image_size)
    generator.summary()

def train(models, x_train, params):
    generator, discriminator, adversarial = models
    batch_size, latent_size, train_steps, model_name = params

    save_interval = 500

    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])  # 16, 100

    train_size = x_train.shape[0]   # 60000

    for i in range(train_steps):    # 40000
        # 데이터셋에서 임의로 진짜 이미지를 선택
        rand_indexes = np.random.randint(0, train_size, size=batch_size)    # 64개
        real_images = x_train[rand_indexes] # (64, 28, 28, 1)

        # 생성기를 사용해 노이즈로부터 가짜 이미지 생성, 균등분포를 사용해 노이즈 생성
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])    # 64, 100
        # 가짜 이미지 생성
        fake_images = generator.predict(noise)  # (64, 28, 28, 1)

        # 진짜 이미지 + 가짜 이미지 = 훈련 데이터의 1 배치(batch)
        x = np.concatenae((real_images, fake_images))
        # 진짜 이미지에는 1, 가짜 이미지에는 0 으로 라벨링
        y = np.ones([2 * batch_size, 1])            # (128, 1)
        y[batch_size:, :] = 0.0
        
        # 판별기 네트워크 훈련. 
        loss, acc = discriminator.train_on_batch(x, y)
        log = "%d : [discriminator loss : %f, acc : %f]" %(i, loss, acc)

        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])    #(64, 100)
        y = np.ones([batch_size, 1])    # (64, 1)   

        # 적대적 네트워크 훈련.
        loss, acc = adversarial.train_on_batch(noise, y)
        log = "%s [adversarial loss : %f, acc : %f]" %(log, loss, acc)

        print(log)


        if (i + 1) % save_interval == 0:
            if(i+1) == train_steps:
                show = True
            else:
                show = False

            plot_images(generator,
                       noise_input = noise_input,
                       show = show,
                       step = (i + 1),
                       model_name = model_name )

    generator.save(model_name + ".h5")                    

def plot_images(generator, 
                noise_input,
                show=False,
                step=0,
                model_name='gen'):

    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict(noise_input)
    plt.figure(figsize=(2.2, 2.2))






















