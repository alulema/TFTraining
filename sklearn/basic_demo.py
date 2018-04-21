import matplotlib.pyplot as plt
import numpy as np
import pylab
from sklearn import datasets
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier

iris = datasets.load_iris();
X_iris, Y_iris = iris.data, iris.target
X, Y = X_iris[:, :2], Y_iris

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)
scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

colors = ['red', 'blue', 'lightgreen']

for i in range(len(colors)):
    xs = X_train[:, 0][Y_train == i]
    ys = X_train[:, 1][Y_train == i]
    plt.scatter(xs, ys, c=colors[i])

plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

clf = SGDClassifier()
clf.fit(X_train, Y_train)

print(clf)
print(clf.coef_)
print(clf.intercept_)

x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)

for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class ' + str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    pylab.sca(axes[i])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.prism)
    ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0]/clf.coef_[i, 1])
    plt.plot(xs, ys, hold=True)

plt.show()

print(clf.predict(scaler.transform([[4.7, 3.1]])))
print(clf.decision_function(scaler.transform([[4.7, 3.1]])))

from sklearn import metrics

Y_train_pred = clf.predict(X_train)
print(metrics.accuracy_score(Y_train, Y_train_pred))

Y_pred = clf.predict(X_test)
print(metrics.accuracy_score(Y_test, Y_pred))

print(metrics.classification_report(Y_test, Y_pred, target_names=iris.target_names))
