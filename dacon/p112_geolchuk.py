import numpy as np
import pandas as pd

df = pd.DataFrame({'a':np.random.choice(['a', 'b', 'c'], size=10),
                   'b':np.random.randn(10),
                   'c':np.random.uniform(size=10)}) 
print(df)
print(df.shape)

# 결측값 생성
df.loc[np.random.choice(np.arange(10), 3), ['b']] = np.nan

print(df)

print(df.isnull())
print(df.isnull().sum())

# print(df.groupby('x').agg('mean'))

df['d'] = (df['b'] + df['c'])
print(df)
# df['a'] = df.mean(df['y'],df['z'])
# df['b'] = df.iloc[:,1:3].mean()
# print(df)

# df['b'] = df.loc[:,['y','z']].mean()

# print(df.iloc[:,[2]])
# print(df.iloc[:,1:3])

# print(df.loc[:,['x']])
# print(df.loc[[1,3,5,7,9],['x']])

# print(df.fillna(method='bfill', axis=1))
print(df.interpolate())




