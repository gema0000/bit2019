#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# 1. load text

# x1 = np.loadtxt('./study/data/evaluate_test.csv', encoding='cp949')
# x2 = np.loadtxt('./study/data/evaluate_test.csv', delimiter=',', dtype=np.int64, encoding='utf-8')
# x3 = np.loadtxt('./study/data/evaluate_test.csv', encoding='euc-kr')
# x = np.loadtxt('./study/data/evaluate_test.csv')

test = pd.read_csv('./study/test/test0822_ljs4.csv', header=None).values
answer = pd.read_csv('./study/test/answer.csv', header=None).values
print(test)
print(type(test))
print(test.shape)
print(answer)
print(type(answer))
print(answer.shape)

# 2. rmse

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(test, answer):
    return np.sqrt(mean_squared_error(test, answer))
print("RMSE : ", RMSE(test, answer))
