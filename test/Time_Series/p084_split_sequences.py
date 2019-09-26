from numpy import array, hstack
from keras.models import Sequential
from keras.layers import Dense

# split a multivariate sequence into samples 
def split_sequences(sequences, n_steps_in, n_steps_out): 
    X, y = list(), list() 
    for i in range(len(sequences)): 
        # find the end of this pattern 
        end_ix = i + n_steps_in 
        out_end_ix = end_ix + n_steps_out 
        # check if we are beyond the dataset 
        if out_end_ix > len(sequences): 
            break 
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :] 
        X.append(seq_x) 
        y.append(seq_y) 
    return array(X), array(y)    

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90]) 
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95]) 
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))]) 

in_seq1 = in_seq1.reshape((len(in_seq1), 1))    #(9, 1)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))    #(9, 1)
out_seq = out_seq.reshape((len(out_seq), 1))    #(9, 1)

dataset = hstack((in_seq1, in_seq2, out_seq)) 

n_steps_in, n_steps_out = 3, 2      # 입력 3,3 개, 출력 2,3 개로 자른다.
X, y = split_sequences(dataset, n_steps_in, n_steps_out) 
print(X)
print("==================================")
print(y)
print(X.shape)  # (5, 3, 3)
print(y.shape)  # (5, 2, 3)
print("======= Flatten input ============")
n_input = X.shape[1] * X.shape[2] 
X = X.reshape((X.shape[0], n_input))
print(X.shape)  # (5, 9)
print("======= Flatten output ============")
# n_output = y.shape((X.shape))






