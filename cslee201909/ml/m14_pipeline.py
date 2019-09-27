import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

# 학습 전용과 테스트 전용 분리하기 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
# from sklearn.pipeline import make_pipeline
# pipe = make_pipeline(MinMaxScaler(), SVC(C=100))

pipe.fit(x_train, y_train)

print("테스트 점수 : ", pipe.score(x_test, y_test))


