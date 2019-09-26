from numpy import array

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

# define input sequence 
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90] 
# choose a number of time steps 
n_steps = 3 
# split into samples 
X, y = split_sequence(raw_seq, n_steps) 
# summarize the data 
for i in range(len(X)): 
    print(X[i], y[i])
'''
from keras.models import Sequential
from keras.layers import Dense
# define model
model = Sequential()    
model.add(Dense(100, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=2000, verbose=1)
'''
# demonstrate prediction
x_input = array([70, 80, 90])
print(x_input.shape)
x_input = x_input.reshape((1, n_steps))
# yhat = model.predict(x_input, verbose=1)
print(x_input.shape)
# print(yhat)
