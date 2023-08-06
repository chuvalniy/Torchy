from abc import ABC
import numpy as np
from layer import NnLayer
from value import Value


class _Optim(ABC):
    def __init__(self, params: dict[str, Value], lr: float = 1e-4, weight_decay: float = 0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        for layer in self.params.values():
            if isinstance(layer, NnLayer):
                layer.zero_grad()


class SGD(_Optim):
    def step(self):
        for param in self.params.values():
            gradient = param.grad + self.weight_decay * 2 * param.data
            param.data -= self.lr * gradient

    def get_regularization(self):
        l2_loss = 0.0
        for param in self.params.values():
            l2_loss += self.weight_decay * np.sum(np.square(param.data))

        return l2_loss
