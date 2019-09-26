from numpy import array, hstack

# split a univariate sequence into samples 
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
X, y = split_sequence(raw_seq, n_steps)
for i in range(len(X)):
    print(X[i], y[i])

print("우갸갸갹")

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                 input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()