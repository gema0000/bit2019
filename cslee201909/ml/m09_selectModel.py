import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
# scikit-learn 0.20.3 에서 31개
# scikit-learn 0.21.2 에서 40개중 4개만 돔.

# pip uninstall scikit-learn
# pip install scikit-learn==0.20.3

import warnings
warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./_data/csv/iris2.csv", encoding="utf-8")

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
print(len(allAlgorithms))   # 31
print(type(allAlgorithms))  # list

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기 --- (*2)
    clf = algorithm()

    # 학습하고 평가하기 --- (*3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name,"의 정답률 = " , accuracy_score(y_test, y_pred))
  
'''
AdaBoostClassifier 의 정답률 =  0.9666666666666667
BaggingClassifier 의 정답률 =  0.9666666666666667
BernoulliNB 의 정답률 =  0.3
CalibratedClassifierCV 의 정답률 =  0.9333333333333333
ComplementNB 의 정답률 =  0.6666666666666666
DecisionTreeClassifier 의 정답률 =  0.9666666666666667
ExtraTreeClassifier 의 정답률 =  0.9666666666666667
ExtraTreesClassifier 의 정답률 =  0.9666666666666667
GaussianNB 의 정답률 =  0.9666666666666667
GaussianProcessClassifier 의 정답률 =  1.0
GradientBoostingClassifier 의 정답률 =  0.9666666666666667
KNeighborsClassifier 의 정답률 =  1.0
LabelPropagation 의 정답률 =  0.9666666666666667
LabelSpreading 의 정답률 =  0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 =  0.9666666666666667
LinearSVC 의 정답률 =  0.9333333333333333
LogisticRegression 의 정답률 =  0.9666666666666667
LogisticRegressionCV 의 정답률 =  0.9666666666666667
MLPClassifier 의 정답률 =  1.0
MultinomialNB 의 정답률 =  0.9666666666666667
NearestCentroid 의 정답률 =  0.9666666666666667
NuSVC 의 정답률 =  1.0
PassiveAggressiveClassifier 의 정답률 =  0.8333333333333334
Perceptron 의 정답률 =  0.6666666666666666
QuadraticDiscriminantAnalysis 의 정답률 =  0.9666666666666667
RadiusNeighborsClassifier 의 정답률 =  1.0
RandomForestClassifier 의 정답률 =  0.9666666666666667
RidgeClassifier 의 정답률 =  0.8333333333333334
RidgeClassifierCV 의 정답률 =  0.8333333333333334
SGDClassifier 의 정답률 =  0.6666666666666666
SVC 의 정답률 =  1.0
'''

