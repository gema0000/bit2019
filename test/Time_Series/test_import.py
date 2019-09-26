import p089_cnn as ti

from numpy import array, hstack

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90] 

n_steps = 3
X, y = ti.split_sequence(raw_seq, n_steps)
for i in range(len(X)):
    print(X[i], y[i])
