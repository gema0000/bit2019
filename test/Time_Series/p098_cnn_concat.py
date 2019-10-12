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
n_features = 1

# convert into input/output 
X, y = split_sequences(dataset, n_steps) 
print(X.shape, y.shape) # (7, 3, 2) (7, )

# summarize the data 
for i in range(len(X)): 
    print(X[i], y[i])

X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], n_features)
X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], n_features)

from keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, concatenate, Input 
from keras.models import Sequential, Model

# first input model
visible1 = Input(shape=(n_steps, n_features))
cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible1)
cnn1 = MaxPooling1D(pool_size=2)(cnn1)
cnn1 = Flatten()(cnn1)

visible2 = Input(shape=(n_steps, n_features))
cnn2 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible2)
cnn2 = MaxPooling1D(pool_size=2)(cnn2)
cnn2 = Flatten()(cnn2)

merge = concatenate([cnn1, cnn2])
dense = Dense(50, activation='relu')(merge)
output = Dense(1)(dense)

model = Model(inputs=[visible1, visible2], outputs=output)
model.compile(optimizer='adam', loss='mse')

model.fit([X1, X2], y, epochs=1000, verbose=1)

x_input = array([[80, 85], [90, 95], [100, 105]])
x1 = x_input[:, 0].reshape((1, n_steps, n_features))
x2 = x_input[:, 1].reshape((1, n_steps, n_features))

yhat = model.predict([x1, x2], verbose=1)
print(yhat)