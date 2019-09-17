
input = Input(shape=(28,28,1))

x = Conv2D(16,(3,3), activation='relu', padding='same')


# UpSampling2D


'''
1. input shape를 28,28,1 로 쭉 가다가
output shape도 28,28,1 로 하면 된다.
2. padding=valid, 또는 maxpooling으로 해서 크기를 줄일경우
decoder 에서는 upSampling2D로 키워준다.
'''

# RandomSearch로 오토인코더를 사용하시오.
# m28_autoencoder2_hyper.py

