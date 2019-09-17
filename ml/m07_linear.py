from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)
print(boston.keys())
print(boston.target)
print(boston.target.shape)

x = boston.data
y = boston.target

print(type(boston)) # <class 'sklearn.utils.Bunch'>

from sklearn.linear_model import LinearRegression #, Ridge, Lasso, 
# 모델 완성하시오.
