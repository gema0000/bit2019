from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model, Sequential
import numpy as np

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

sequence = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# reshape input into [samples, timesteps, features]
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))
print(sequence)                             # (1,9,1)

model = Sequential()

model.add(LSTM(100, activation='relu', batch_input_shape=(1,n_in,1), stateful=True,
                    return_sequences=True ))    # (1,9,1)
model.add(RepeatVector(n_in))
# model.add(LSTM(100, activation='relu', return_sequences=True, stateful=True))
model.add(LSTM(100, activation='relu', return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(1)))

model.compile(optimizer='adam', loss='mse')

model.summary()

num_epochs = 10
for epoch_idx in range(num_epochs):
	print('epochs : '+ str(epoch_idx) )
	# model.fit(sequence, sequence, epochs=1, batch_size=1, verbose=1, shuffle=False)
	model.fit(sequence, sequence, epochs=1, verbose=1, shuffle=True)
	model.reset_states()

print('--------')
eval = model.evaluate(sequence, sequence)
print(eval)
