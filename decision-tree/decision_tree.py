from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# Decision Tree Classification
iris = load_iris()
x = iris.data[:, 2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(x, y)

x_test = [[5, 1.5]]
y_test = tree_clf.predict(x_test)
y_test_prob = tree_clf.predict_proba(x_test)
print(f"Predicted Class Probabilities: {y_test_prob[0]}")

plt.scatter(x[:, 0], x[:, 1], cmap='jet', c=y)
plt.scatter(x_test[0][0], x_test[0][1], s=100, label=f"Predicted Class: {int(y_test)}")
plt.title("Iris Flower Data")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
plt.show()

# Feature importance
print("Iris Flower Feature Importance")
for name, score in zip(iris['feature_names'], tree_clf.feature_importances_):
    print("\t", name, score)


# Decision Tree Regression
m = 100
x = 2 * np.random.rand(m, 1) + 3
y = 3 * x + np.random.randn(m, 1)

plt.scatter(x, y)

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(x, y)

x_test = [[4.5]]
y_test = tree_reg.predict(x_test)

plt.scatter(x_test, y_test, s=100, label=f"Decision Tree Prediction for x = {x_test[0][0]}")
plt.legend()
plt.show()


# Accuracy of Various Models on Make_Moons Data
x, y = make_moons(1000, True, noise=0.30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
plt.scatter(x_train[:, 0], x_train[:, 1], cmap='jet', c=y_train)
plt.title("Make_Moons Data")
plt.show()


# Voting Classifier Ensemble
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
svm_clf = SVC(probability=True)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft'
)
voting_clf.fit(x_train, y_train)

# Measure accuracy
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# Bagging with Out of Bag (OOB) score

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True)
bag_clf.fit(x_train, y_train)
y_pred = bag_clf.predict(x_test)
print(f"Out of Bag Score {bag_clf.oob_score_}")
print(bag_clf.__class__.__name__, accuracy_score(y_test, y_pred))


# AdaBoost
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm='SAMME.R', learning_rate=0.5)
ada_clf.fit(x_train, y_train)
print(ada_clf.__class__.__name__, accuracy_score(y_test, y_pred))


# Gradient Boosting
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(x_train, y_train)
print(gbrt.__class__.__name__, accuracy_score(y_test, y_pred))


# Choosing number of estimators for the most accurate Gradient Boosting Regressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(x_train, y_train)
errors = [mean_squared_error(y_test, y_pred)
          for y_pred in gbrt.staged_predict(x_test)]
best_n_estimators = np.argmax(errors) + 1

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)
gbrt_best.fit(x_train, y_train)
print(gbrt_best.__class__.__name__, "Best", accuracy_score(y_test, y_pred))
