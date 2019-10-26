from keras.layers import Input, Embedding, merge, dot, Dense
from keras.models import Model, Sequential

# model = Model([user_in, movie_in], x);

n_users = 10
n_movies = 10
n_factors = 2

user_in = Input(shape=(1,), dtype='int64', name='user_in')
u = Embedding(n_users, n_factors, input_length=1)(user_in)
movie_in = Input(shape=(1,), dtype='int64', name='movie_in')
v = Embedding(n_movies, n_factors, input_length=1)(movie_in)

# x = merge([u, v], mode='dot')
x = dot([u, v], axes=-1)

x = Dense(10)(x)
x = Dense(10)(x)
x = Dense(10)(x)

model = Model(inputs = [user_in, movie_in], outputs = x)

model.summary()

'''
from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_2, dense2_3])

middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(7)(middle2)

######################### 요기부터 아웃풋 모델

output1 = Dense(30)(middle3)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)    # 33 -> 3

output2 = Dense(20)(middle3)
output2_2 = Dense(70)(output2)
output2_3 = Dense(3)(output2_2)

model = Model(inputs = [input1, input2],
              outputs = [output1_3, output2_3]  
)
model.summary()
'''