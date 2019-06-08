"""
This code is a verbatim implementation of 'A Novel Chaos Theory Inspired Neuronal Architecture'
by Harikrishnan N B and Nithin Nagaraj: https://arxiv.org/pdf/1905.12601.pdf
"""
from scipy.spatial.distance import cosine
import numpy as np
import tqdm

class GLSNeuron(object):
    def __init__(self, q=0.5, b=0.55, epsilon=0.05):
        """
        Initial q and b are based on 
        https://www.machinecurve.com/index.php/2019/06/02/could-chaotic-neurons-reduce-machine-learning-data-hunger/#testing-on-the-mnist-dataset
        """        
        self.b = b
        self.q = q
        self.epsilon = epsilon

    def T_map(self, x):
        """Compute  Generalized Luroth Series map function."""
        if x >= 0 and x < self.b:
            return x/self.b
        elif self.b <= x and x < 1:
            return (1 - x)/(1 - self.b)
        else:
            ValueError("invalid value encountered {}".format(x))

    def __call__(self, x):
        """Compute firing time."""
        w = self.q
        I_high, I_low = x + self.epsilon, x - self.epsilon 
        N_it = 0
        N_tol = 1000
        while True:
            w = self.T_map(w)
            N_it += 1
            if w <= I_high and w >= I_low:
                return N_it
            if N_it == N_tol:
                raise ValueError("Neuron iterated {} times. \nValue: {}".format(N_it, w))
                


class GLSLayer(object):
    def __init__(self, q, b, epsilon):
        """Both b and q are vectors for initial q and b values for all neurons, epsilon is a constant."""
        self.q = q
        self.b = b
        self.epsilon = epsilon
        self.neurons = []
        self.M = None
        self.populate_layer()

    def populate_layer(self):
        """Create a list of neurons."""
        assert len(self.q) == len(self.b)
        for i in range(len(self.q)):
            qi, bi = self.q[i], self.b[i]
            n = GLSNeuron(qi, bi, self.epsilon)
            self.neurons.append(n)

    def normalize(self, X):
        """Normalize data matrix to be between 0 and 1."""
        min_X = np.min(X)
        X_ = X - min_X
        return X_ / (np.max(X) - min_X)

    def extract_features(self, X):
        """
        Go through X and apply the layer to extract features. 
        (Algorithm 2. in the paper)
        """
        X = self.normalize(X)
        M = np.zeros(X.shape)
        for i, sample in enumerate(X):
            for j, value in enumerate(sample):
                neuron = self.neurons[j]
                M[i, j] = neuron(value)
        return M

    def fit(self, X, y):
        """
        Extract 'prototype vectors' for all classes. 
        (Algorithm 1. in the paper)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        X = self.normalize(X)
        self.classes = list(set(y))
        self.n_classes = len(self.classes)
        M = []
        for cl in self.classes:
            inds = np.where(y == cl)[0]
            Xs = X[inds,:]
            M.append(self.extract_features(Xs).mean(axis=0))
        self.M = np.asarray(M)
                
    def predict(self, X):
        """
        Go through all samples and predict class. 
        (Algorithm 3. in the paper)
        """
        Z = self.normalize(X)
        F = self.extract_features(Z)
        y_hat = np.zeros(F.shape[0])
        for i, sample in enumerate(F):
            max_sim = 0
            max_c = None
            for c, prototype in enumerate(self.M):
                sim = 1 - cosine(sample, prototype)
                if sim >= max_sim:
                    max_sim = sim
                    max_c = c
            y_hat[i] = self.classes[max_c]
        return y_hat




        
