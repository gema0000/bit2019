import numpy as np

def train(models, x_train, params):
    generator, discriminator, adversarial = models

    batch_size, latent_size, train_steps, model_name = params
    # 64, 100, 40000, "dcgan_mnist"

    # 500 단계마다 생성기 이미지가 저장됨
    save_interval = 500
    # 훈련 기간 동안 생성기 출력 이미지가 어떻게 진화하는 지 보기 위한 노이즈 벡터
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])    

    # 훈련 데이터세트에 포함된 요소 수
    train_size = x_train.shape[0]   # 50000?
    for i in range(train_steps):
        # 데이터세트에서 임의로 진짜 이미지를 선택
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]

        # 생성기를 사용해 노이즈로부터 가짜 이미지 생성
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_images = generator.predict(noise)

        # 진짜 이미지 + 가짜 이미지 = 훈련 데이터의 1 배치(batch))

        x = np.concatenate((real_images, fake_images))

        # 진짜와 가짜 이미지에 레이블을 붙임
        # 진짜 이미지의 레이블은 1.0
        y = np.ones([2 * batch_size, 1])

        # 가짜 이미지의 레이블은 0.0
        y[batch_size:, :] = 0.0

        # 판별기 네트워크 훈련, 손실과 정확도 기록(log)
        loss, acc = discriminator.train_on_batch(x, y)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

            











