from numpy import array, hstack

# # split a univariate sequence into samples 
# def split_sequence(sequence, n_steps): 
#     X, y = list(), list() 
#     for i in range(len(sequence)): 
#         # find the end of this pattern 
#         end_ix = i + n_steps 
#         # check if we are beyond the sequence 
#         if end_ix > len(sequence)-1:
#             break 
#         # gather input and output parts of the pattern 
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] 
#         X.append(seq_x) 
#         y.append(seq_y) 
#     return array(X), array(y)

# split a multivariate sequence into samples 
def split_sequences(sequences, n_steps): 
    X, y = list(), list() 
    for i in range(len(sequences)): 
        # find the end of this pattern 
        end_ix = i + n_steps 
        # check if we are beyond the sequence 
        if end_ix > len(sequences):
            break 
        # gather input and output parts of the pattern 
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1] 
        X.append(seq_x) 
        y.append(seq_y) 
        # print(seq_x, "///", seq_y)
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
print(dataset.shape)
print("========================")
n_steps = 3
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)

for i in range(len(X)):
    print(X[i], y[i])

# p.061
n_input = X.shape[1] * X.shape[2]
print(n_input)
X = X.reshape((X.shape[0], n_input))
print(X)
print(y)

from keras.models import Sequential
from keras.layers import Dense
# define model
model = Sequential()    
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=2000, verbose=1)

# demonstrate prediction
x_input = array([[80,85], [90,95], [100,105]])
x_input = x_input.reshape((1, n_input))

yhat = model.predict(x_input, verbose=1)
# print(x_input.shape)
print(yhat)