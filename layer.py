import numpy as np
from value import Value
import copy
from abc import ABC


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
        self.W = Value(0.001 * np.random.randn(n_input, n_output))
        self.B = Value(0.001 * np.random.randn(1, n_output))
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
