sequence = ([10,20,30,40,50,60,70,80,90])

from numpy import array

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
          break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

n_steps = 3
n_features = 1
# reshape from [samples, timesteps] into [samples, timesteps, features]
X, y = split_sequence(sequence, n_steps)

print(X.shape)
X = X.reshape((X.shape[0], X.shape[1], n_features))
print(X.shape)

for i in range(len(X)):
    print(X[i], y[i])

from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, LSTM
from keras.models import Sequential

model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
model.add(LSTM(64, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=1000, verbose=1, batch_size=1)

x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=1)

print(yhat)





