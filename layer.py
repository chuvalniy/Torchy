import copy
from abc import ABC

import numpy as np

from value import Value


class Layer(ABC):

    def backward(self, d_out):
        pass

    def forward(self, X):
        pass

    def __call__(self, X):
        return self.forward(X)


class NnLayer(Layer):
    def zero_grad(self):
        pass

    def params(self) -> dict[str, Value]:
        pass


class Linear(NnLayer):
    def __init__(self, n_input, n_output):
        self.W = Value(np.sqrt(2) / np.sqrt(n_input) * np.random.normal(size=(n_input, n_output)))
        self.B = Value(1e-3 * np.random.normal(size=(1, n_output)))
        self.X = None

    def forward(self, X):
        self.X = copy.deepcopy(X)

        return np.dot(self.X, self.W.data) + self.B.data

    def backward(self, d_out):
        self.W.grad = np.dot(self.X.T, d_out)

        E = np.ones(shape=(1, self.X.shape[0]))
        self.B.grad = E.dot(d_out)

        d_pred = np.dot(d_out, self.W.data.T)

        return d_pred

    def zero_grad(self):
        for _, value in self.params().items():
            value.grad = 0.0

    def params(self) -> dict[str, Value]:
        d = {
            "W": self.W,
            "B": self.B
        }

        return d


class ReLU(Layer):
    def __init__(self):
        self._mask = None

    def forward(self, X):
        self._mask = X > 0
        return np.where(self._mask, X, 0)

    def backward(self, d_out):
        return d_out * self._mask


class BatchNorm1d(NnLayer):
    def __init__(self, n_output, eps=1e-5):
        self.out = None
        self.X_norm = None
        self.X_var = None
        self.X_mean = None
        self.X = None
        self.eps = eps
        self.gamma = Value(np.ones(n_output))
        self.beta = Value(np.zeros(n_output))

    def forward(self, X):
        self.X = copy.deepcopy(X)

        self.X_mean = np.mean(X, axis=0, keepdims=True)
        self.X_var = np.var(X, axis=0, keepdims=True)
        self.X_norm = (X - self.X_mean) / np.sqrt(self.X_var + self.eps)

        self.out = self.gamma.data * self.X_norm + self.beta.data
        return self.out

    def backward(self, d_out):
        self.gamma.grad = (self.X_norm * d_out).sum(axis=0)
        self.beta.grad = d_out.sum(0)

        batch_size = self.X.shape[0]

        sqrt_var_eps = np.sqrt(self.X_var + self.eps)
        dxhat = d_out * self.gamma.data
        dvar = np.sum(dxhat * (self.X - self.X_mean), axis=0) * (-1 / 2) * (self.X_var + self.eps) ** (-3 / 2)
        dmu = np.sum(dxhat * (-1 / sqrt_var_eps), axis=0) + dvar * (-2 / batch_size) * np.sum(self.X - self.X_mean, axis=0)
        dx = dxhat * (1 / sqrt_var_eps) + dvar * (2 / batch_size) * (self.X - self.X_mean) + dmu / batch_size
        return dx

    def params(self) -> dict[str, Value]:
        d = {
            "gamma": self.gamma,
            "beta": self.beta
        }

        return d
