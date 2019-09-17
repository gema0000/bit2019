import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
# scikit-learn 0.20.3 에서 31개
# scikit-learn 0.21.2 에서 40개중 4개만 돔.

import warnings
warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기 
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

# 학습 전용과 테스트 전용 분리하기 
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                    train_size = 0.8, shuffle = True)

# classifier 알고리즘 모두 추출하기--- (*1)
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

print(allAlgorithms)
print(len(allAlgorithms))
print(type(allAlgorithms))

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기 --- (*2)
    clf = algorithm()

    # 학습하고 평가하기 --- (*3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name,"의 정답률 = " , accuracy_score(y_test, y_pred))
  