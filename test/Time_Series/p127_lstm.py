from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense

def split_sequence(sequence, n_steps): 
    X, y = list(), list() 
    for i in range(len(sequence)): 
        # find the end of this pattern 
        end_ix = i + n_steps 
        # check if we are beyond the sequence 
        if end_ix > len(sequence)-1: 
            break 
        # gather input and output parts of the pattern 
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] 
        X.append(seq_x) 
        y.append(seq_y) 
    return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90] 

n_steps = 3
n_features = 1

X, y = split_sequence(raw_seq, n_steps)

print(X)
print(y)

X = X.reshape((X.shape[0], X.shape[1], n_features))

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=1000, verbose=0)

x_input = array([70,80,90])
x_input = x_input.reshape((1, n_steps, n_features))

y_hat = model.predict(x_input, verbose=0)
print(y_hat)






