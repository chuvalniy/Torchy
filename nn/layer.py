import numpy as np
from value import Value
import copy
from abc import ABC


class Module(ABC):

    def backward(self, d_out):
        pass

    def forward(self, X):
        pass

    def __call__(self, X):
        return self.forward(X)


class NnModule(Module):
    def zero_grad(self):
        pass

    def params(self):
        pass


class Linear(NnModule):
    def __init__(self, n_input, n_output):
        self.W = Value(n_input, n_output)
        self.B = Value(1, n_output)

        self.X = None

    def forward(self, X):
        self.X = copy.deepcopy(X)

        return np.dot(self.X, self.W.data) + self.B.data

    def backward(self, d_out):
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=1)

        d_pred = np.dot(self.W.data, d_out.T)

        return d_pred

    def zero_grad(self):
        for _, value in self.params().items():
            value.grad = 0.0

    def params(self):
        d = {
            "W": self.W,
            "B": self.B
        }

        return d


class ReLU(Module):
    def __init__(self):
        pass

    def __call__(self, X):
        return X > 0

    def backward(self, d_out):
        pass
