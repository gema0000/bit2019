import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index={}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index)+1
max_length=10

results = np.zeros(shape=(len(samples), max_length, max(token_index.values())+1))
print(results)

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1
    
aaa = list(zip([1,2,3,4,5], [4,5,6,8]))
print(aaa)
print(type(aaa))


