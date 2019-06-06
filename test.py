from sklearn.datasets import load_iris
import numpy as np
from neuron import GLSLayer


iris = load_iris()
q = np.ones(len(iris.target)) * 0.24
b = np.ones(len(iris.target)) * 0.467354
epsilone = 0.05

layer = GLSLayer(q, b, epsilone)
print(layer.extract_features(iris.data))
layer.fit(iris.data, iris.target)
print(layer.predict(iris.data))
