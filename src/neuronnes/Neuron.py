import numpy as np


class Neuron:

    def __init__(self, n_inputs=2):
        self.weights = np.random.randn(n_inputs)
        self.bias = 0.0

    def aggregation(self, X):
        return X @ self.weights + self.bias

    def activation(self, z):
        # sigmoide
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        return self.activation(self.aggregation(X))

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

    def loss(self, a, y):
        # clip sinon log(0)
        a = np.clip(a, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))

    def gradients(self, X, y, a=None):
        # dL/dz = a - y, a peut etre fourni pour eviter un second forward
        if a is None:
            a = self.forward(X)
        error = a - y
        dw = X.T @ error / len(y)
        db = np.mean(error)
        return dw, db

    def update(self, dw, db, learning_rate):
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
