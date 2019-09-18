from keras.layers import Input, Dense, concatenate, Concatenate
from keras.models import Model, Sequential

inputs = Input(shape = (10,))
x = Dense(9)(inputs)
x = Dense(3)(x)
x = Dense(1)(x)

model01 = Model(inputs, x)

model02 = Sequential()
model02.add(Dense(4, input_dim=3 ))
model02.add(Dense(1))

merged = Concatenate([model01, model02])

# result = Sequential()
# result.add(merged)
# result.summary()

# merged.summary()




