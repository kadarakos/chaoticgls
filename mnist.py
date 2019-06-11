from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
from neuron import GLSLayer
from keras.datasets import mnist
import gzip
import _pickle as cPickle
import tqdm

with gzip.open('/roaming/u1257964/MNIST/mnist.pkl.gz') as f:
    mnist = cPickle.load(f, encoding='latin1') 

(X, y), (_, _) = mnist
X = X.reshape(60000, 784)
print(X.shape, y.shape)
best = 0
f1s = []

for q in np.linspace(0.3, 0.9, 50):
    for b in np.linspace(0.3, 0.9, 50):
        if q == b:
            continue
        Q = np.ones(X.shape[1]) * q
        B = np.ones(X.shape[1]) * b
        epsilone = 0.05
        layer = GLSLayer(Q, B, epsilone, parallel=True)

        skf = StratifiedKFold(n_splits=10, random_state=620)
        skf.get_n_splits(X, y)
        F1 = []
        for train_index, test_index in tqdm.tqdm(list(skf.split(X, y))):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            layer.fit(X_train, y_train)
            y_hat = layer.predict(X_test)
            f1 = f1_score(y_test, y_hat, average='macro')
            F1.append(f1)
        score = np.mean(F1)
        f1s.append(score)
        F1 = []
        print(score)
        if score > best:
            print(q, b, score)
            best = score

with open('f1s.pkl', 'w') as f:
    for item in f1s:
        f.write("%s\n" % item)
