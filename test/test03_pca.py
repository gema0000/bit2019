# 파이썬 라이브러리를 활용한 머신러닝 P.198
# 차원축소 PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print("원본 데이터 형태 : ", str(X_scaled.shape))
print("축소된 데이터 형태 : ", str(X_pca.shape))

