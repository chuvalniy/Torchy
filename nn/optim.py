from abc import ABC

from layer import NnModule


class _Optim(ABC):
    def __init__(self, params: dict, lr: float = 1e-1, reg: float = 1e1):
        self.params = params
        self.lr = lr
        self.reg = reg

    def zero_grad(self):
        for _, layer in self.params.items():
            if isinstance(layer, NnModule):
                layer.zero_grad()
