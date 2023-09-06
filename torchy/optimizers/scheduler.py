from abc import ABC

import numpy as np

from .optim import _Optim


class _Scheduler(ABC):
    """
    Abstract class for scheduler algorithm.
    """

    def __init__(self, optimizer: _Optim, verbose: bool = False):
        """
        :param optimizer: _Optim - gradient descent optimizer.
        :param verbose: bool - on/off logging every scheduler step.
        """
        self.optimizer = optimizer
        self.verbose = verbose

    def step(self, *args, **kwargs):
        """
        Abstract method for scheduler step.

        :param args:
        :param kwargs:
        """
        pass


class StepLR(_Scheduler):
    """
    Decays learning rate of optimizer every self.step_size by self.gamma.
    """

    def __init__(self, optimizer: _Optim, step_size: int, gamma: float = 1e-1, verbose: bool = False):
        """
        :param optimizer:  _Optim - gradient descent optimizer.
        :param step_size: int - number of steps until decaying learning rate.
        :param gamma: float - learning rate decay factor.
        :param verbose: on/off logging every scheduler step.
        """
        super().__init__(optimizer=optimizer, verbose=verbose)
        self.step_size = step_size
        self.gamma = gamma
        self._epoch = 0

    def step(self):
        """
        Counts number of epochs and decays learning rate based on the current number of epochs and step_size.
        """
        self._epoch += 1

        if self._epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            if self.verbose:
                print(f"Epoch #{self._epoch} learning rate updated to {self.optimizer.lr}")


class ReduceLROnPlateau(_Scheduler):
    """
    Decays learning rate of optimizer when a metric has stopped improvement.
    """

    def __init__(self, optimizer: _Optim, factor: float = 1e-1, patience: int = 5, verbose: bool = False,
                 threshold: float = 1e-4):
        """
        :param optimizer: _Optim - gradient descent optimizer.
        :param factor: float - learning rate decay factor.
        :param patience: int - number of iterations without improvement.
        :param verbose: bool - on/off logging every scheduler step.
        :param threshold: float - measurement to focus only on significant improvements.
        """
        super().__init__(optimizer=optimizer, verbose=verbose)
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.wait = 0
        self.best_loss = np.inf

    def step(self, loss: float):
        """
        Increases self.wait by 1 if error is not improving and decays learning rate.
        of the optimizer of self.wait >= self.patience.

        :param loss: current values of the model error.
        """
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
