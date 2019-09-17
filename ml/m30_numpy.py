import numpy as np
a = np.arange(10)
print(a)
np.save("aaa.npy", a)
b = np.load("aaa.npy")
print(b)
