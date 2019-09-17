# 결정트리의 단점 : 과대적합
# 장점 : 전처리가 필요없다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

# tree = DecisionTreeClassifier(max_depth=4, random_state=0)
# tree.fit(X_train, y_train)
# print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
# print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

print("특성 중요도:\n", tree.feature_importances_)
    