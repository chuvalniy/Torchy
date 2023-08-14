from abc import ABC

import numpy as np

from torchy.layer import NnLayer
from torchy.value import Value


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


class MomentumSGD(_Optim):
    def __init__(self, params, lr: float = 1e-4, weight_decay: float = 0.0, momentum: float = 0.9):
        super().__init__(params=params, lr=lr, weight_decay=weight_decay)
        self.momentum = momentum
        self._velocities = {}

    def _update(self, param_name: str, param: Value):
        velocity = self._velocities.get(param_name, np.zeros_like(param.data))
        gradient = param.grad + self.weight_decay * 2 * param.data

        self._velocities[param_name] = self.momentum * velocity - self.lr * gradient
        param.data += self._velocities[param_name]


class Adagrad(_Optim):
    def __init__(self, params, lr: float = 1e-2, weight_decay: float = 0.0):
        super().__init__(params=params, lr=lr, weight_decay=weight_decay)
        self._accumulated = {}

    def _update(self, param_name: str, param: Value):
        grad_squared = np.square(param.grad)
        grad_accumulated = self._accumulated.get(param_name, np.zeros_like(param.grad))
        self._accumulated[param_name] = grad_accumulated + grad_squared

        adaptive_lr = self.lr / np.sqrt(self._accumulated[param_name])

        param.data -= adaptive_lr * param.grad


class RMSProp(_Optim):
    def __init__(self, params, lr: float = 1e-2, weight_decay: float = 0.0, rho: float = 0.9):
        super().__init__(params=params, lr=lr, weight_decay=weight_decay)
        self.rho = rho
        self._accumulated = {}

    def _update(self, param_name: str, param: Value):
        grad_squared = np.square(param.grad)
        grad_accumulated = self._accumulated.get(param_name, np.zeros_like(param.grad))
        self._accumulated[param_name] = self.rho * grad_accumulated + (1 - self.rho) * grad_squared

        adaptive_lr = self.lr / np.sqrt(self._accumulated[param_name])

        param.data -= adaptive_lr * param.grad


class Adam(_Optim):
    def __init__(self, params, beta1: float = 0.9, beta2: float = 0.999, lr: float = 1e-2, weight_decay: float = 0.0):
        super().__init__(params=params, lr=lr, weight_decay=weight_decay)

        self._velocities = {}
        self._accumulated = {}
        self.beta1 = beta1
        self.beta2 = beta2

    def _update(self, param_name: str, param: Value):
        velocity = self._velocities.get(param_name, np.zeros_like(param.data))
        self._velocities[param_name] = self.beta1 * velocity + (1 - self.beta1) * param.grad

        grad_squared = np.square(param.grad)
        accumulated = self._accumulated.get(param_name, np.zeros_like(param.data))
        self._accumulated[param_name] = self.beta2 + accumulated + (1 - self.beta2) * grad_squared

        adaptive_lr = self.lr / np.sqrt(self._accumulated[param_name])

        param.data -= adaptive_lr * self._velocities[param_name]
