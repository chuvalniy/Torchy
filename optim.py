from abc import ABC

from layer import NnLayer
from value import Value


class _Optim(ABC):
    def __init__(self, params: dict[str, Value], lr: float = 1e-1, reg: float = 0):
        self.params = params
        self.lr = lr
        self.reg = reg

    def zero_grad(self):
        for _, layer in self.params.items():
            if isinstance(layer, NnLayer):
                layer.zero_grad()


class SGD(_Optim):
    def step(self):
        for _, param in self.params.items():
            param.data = param.data - param.grad * self.lr
