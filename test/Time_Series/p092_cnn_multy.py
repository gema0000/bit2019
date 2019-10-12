from numpy import array, hstack

# split a multivariate sequence into samples 
def split_sequences(sequences, n_steps): 
    X, y = list(), list() 
    for i in range(len(sequences)): 
        # find the end of this pattern 
        end_ix = i + n_steps 
        # check if we are beyond the dataset 
        if end_ix > len(sequences): 
            break 
        # gather input and output parts of the pattern 
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1] 
        X.append(seq_x) 
        y.append(seq_y) 
    return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90]) 
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95]) 
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

print(out_seq)

# convert to [rows, columns] structure 
in_seq1 = in_seq1.reshape((len(in_seq1), 1)) 
in_seq2 = in_seq2.reshape((len(in_seq2), 1)) 
out_seq = out_seq.reshape((len(out_seq), 1)) 
# horizontally stack columns 
dataset = hstack((in_seq1, in_seq2, out_seq))

print(dataset)
print(dataset.shape)    # (9, 3)

n_steps = 3
n_features = 2

# convert into input/output 
X, y = split_sequences(dataset, n_steps) 
print(X.shape, y.shape) # (7, 3, 2) (7, )

# summarize the data 
for i in range(len(X)): 
    print(X[i], y[i])

from keras.layers import Conv1D, Flatten, Dense, MaxPooling1D 
from keras.models import Sequential

# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=1000, verbose=1)
# demonstrate prediction

x_input = array([[80, 85], [90, 95], [100, 105]])
print(x_input.shape)
x_input = x_input.reshape((1, n_steps, n_features))
print(x_input.shape)

yhat = model.predict(x_input, verbose=1)
print(yhat)