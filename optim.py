from abc import ABC

import numpy as np

from layer import NnLayer
from value import Value


class _Optim(ABC):
    def __init__(self, params: dict[str, Value], lr: float = 1e-4, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        for layer in self.params.values():
            if isinstance(layer, NnLayer):
                layer.zero_grad()

    def _update(self, param_name: str, param: Value):
        pass

    def step(self):
        for param_name, param in self.params.items():
            self._update(param_name, param)


class SGD(_Optim):
    def _update(self, param_name: str, param: Value):
        gradient = param.grad + self.weight_decay * 2 * param.data
        param.data -= self.lr * gradient


# для каждого параметра сделать свой апдейт по velocity. т.е. по идее просто словарь где ключ имя а значение velocity

class MomentumSGD(_Optim):
    def __init__(self, params, weight_decay: float = 0.0, lr: float = 1e-4, momentum: float = 0.9):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.velocities = {}

    def _update(self, param_name: str, param: Value):
        velocity = self.velocities.get(param_name, np.zeros_like(param.data))
        gradient = param.grad + self.weight_decay * 2 * param.data

        self.velocities[param_name] = self.momentum * velocity - self.lr * gradient
        param.data += self.velocities[param_name]
