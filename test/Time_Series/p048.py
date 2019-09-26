# load time series dataset
import pandas as pd
# series = pd.read_csv('filename.csv', header=0, index_col=0)

from numpy import array
data = list()
n = 5000
for i in range(n):
    data.append([i+1, (i+1)*10])
data = array(data)
print(data[:5, :])
print(data.shape) 
print(type(data))

# drop time
data = data[:, 1]
samples = list()
length = 200

for i in range(0, n, length):
    # grab from i to i + 200
    sample = data[i:i+length]
    samples.append(sample)
# print(samples)
print(len(samples))
print(type(samples))
data = array(samples)
print(data.shape)   #(25, 200)
data = data.reshape((len(samples), length, 1))
print(data.shape)

