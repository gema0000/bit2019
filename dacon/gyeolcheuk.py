import numpy as np #데이터 전처리
import pandas as pd #데이터 전처리
from pandas import DataFrame #데이터 전처리 

import matplotlib.pyplot as plt #데이터 시각화
import seaborn as sns #데이터 시각화

train = pd.read_csv('./_data/dacon/train.csv')
test = pd.read_csv('./_data/dacon/test.csv')
submission = pd.read_csv('./_data/dacon/submission_1002.csv')
print('train shape: ',train.shape)
print('test shape: ',test.shape)
print('submission shape: ',submission.shape)


print(test.isnull().sum())

print(test.interpolate())


