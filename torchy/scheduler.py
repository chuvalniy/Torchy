from abc import ABC

import numpy as np

from torchy.optim import _Optim


class _Scheduler(ABC):
    def __init__(self, optimizer: _Optim, verbose=False):
        self.optimizer = optimizer
        self.verbose = verbose

    def step(self, *args, **kwargs):
        pass


class StepLR(_Scheduler):
    def __init__(self, optimizer: _Optim, step_size: int, gamma: float = 1e-1, verbose=False):
        super().__init__(optimizer=optimizer, verbose=verbose)
        self.step_size = step_size
        self.gamma = gamma
        self._epoch = 0

    def step(self):
        self._epoch += 1

        if self._epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            if self.verbose:
                print(f"Epoch #{self._epoch} learning rate updated to {self.optimizer.lr}")


class ReduceLROnPlateau(_Scheduler):
    def __init__(self, optimizer: _Optim, factor=1e-1, patience=5, verbose=False, threshold=1e-4):
        super().__init__(optimizer=optimizer, verbose=verbose)
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.wait = 0
        self.best_loss = np.inf

    def step(self, loss):
        if loss < self.best_loss - self.threshold:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.optimizer.lr *= self.factor

                if self.verbose:
                    print(f"Learning rate updated to {self.optimizer.lr}")
                self.wait = 0
