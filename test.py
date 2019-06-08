from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
from neuron import GLSLayer


iris = load_iris()
X, y = iris.data, iris.target

best = 0
for q in np.linspace(0.3, 0.9, 50):
    for b in np.linspace(0.3, 0.9, 50):
        if q == b:
            continue
        Q = np.ones(len(iris.target)) * q
        B = np.ones(len(iris.target)) * b
        epsilone = 0.05
        layer = GLSLayer(Q, B, epsilone)

        skf = StratifiedKFold(n_splits=10, random_state=620)
        skf.get_n_splits(X, y)
        F1 = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            layer.fit(X_train, y_train)
            y_hat = layer.predict(X_test)
            F1.append(f1_score(y_test, y_hat, average='macro'))
        score = np.mean(F1)
        F1 = []
        if score > best:
            print(q, b, score)
            best = score
