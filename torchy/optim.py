from abc import ABC

import numpy as np

from torchy.value import Value


class _Optim(ABC):
    """
    Abstract class for all gradient optimization techniques.
    """

    def __init__(self, params: dict[str, Value], lr: float = 1e-4, weight_decay: float = 0.0):
        """
        :param params: dict (parameter_name, parameter) - contains all neural network parameters wrapped
        into dictionary where key is parameter name in a specific layer and value is weight/bias with its own
        data and gradient of that specific layer.
        :param lr: float - learning rate of a neural network model.
        :param weight_decay: float - regularization parameter.
        """
        self._params = params
        self.lr = lr
        self.weight_decay = weight_decay / 2

    def zero_grad(self):
        """
        Goes through the self.params and zeroes parameter gradients.
        """
        for param in self._params.values():
            param.grad = np.zeros_like(param.grad)

    def _update(self, param_name: str, param: Value):
        """
        Single parameter update.

        :param param_name: str - parameter name.
        :param param: Value - holds parameter state (data and gradient).
        """
        pass

    def step(self):
        """
        Loops over parameter dictionary, unpacks it by key and value and then passes
        key and value to the update method.
        """
        for param_name, param in self._params.items():
            self._update(param_name, param)


class SGD(_Optim):
    """
    Implementation of classic Stochastic Gradient Descent.
    """

    def _update(self, param_name: str, param: Value):
        """
        Single SGD parameter update with regularization.

        :param param_name: str - parameter name.
        :param param: Value - holds parameter state (data and gradient).
        """
        gradient = param.grad + self.weight_decay * param.data
        param.data -= self.lr * gradient


class MomentumSGD(_Optim):
    """
    Stochastic Gradient Descent with Momentum
    """

    def __init__(self, params: dict[str, Value], lr: float = 1e-4, weight_decay: float = 0.0, momentum: float = 0.9):
        """
        :param params: dict (parameter_name: str, parameter: Value) - contains all neural network parameters wrapped
        into dictionary where key is parameter name in a specific layer and value is weight/bias with its own
        data and gradient of that specific layer.
        :param lr: float - learning rate of a neural network model.
        :param weight_decay: float - regularization parameter.
        :param momentum: float (0.9-0.999) - momentum parameter (default = 0.9).
        """
        super().__init__(params=params, lr=lr, weight_decay=weight_decay)
        self.momentum = momentum
        self._velocities = {}

    def _update(self, param_name: str, param: Value):
        """
        Single MomentumSGD parameter update with regularization.

        :param param_name: str - parameter name.
        :param param: Value - holds parameter state (data and gradient).
        """
        velocity = self._velocities.get(param_name, np.zeros_like(param.data))
        gradient = param.grad + self.weight_decay * param.data

        self._velocities[param_name] = self.momentum * velocity - self.lr * gradient
        param.data += self._velocities[param_name]


class Adagrad(_Optim):
    """
    Adagrad neural network optimization.
    """

    def __init__(self, params: dict[str, Value], lr: float = 1e-2, weight_decay: float = 0.0, eps: float = 1e-8):
        """
        :param params: dict (parameter_name, parameter) - contains all neural network parameters wrapped
        into dictionary where key is parameter name in a specific layer and value is weight/bias with its own
        data and gradient of that specific layer.
        :param lr: float - learning rate of a neural network model.
        :param weight_decay: float - regularization parameter.
        :param eps: float - value added to numerical stability in the denominator.
        """
        super().__init__(params=params, lr=lr, weight_decay=weight_decay)
        self.eps = eps
        self._accumulated = {}

    def _update(self, param_name: str, param: Value):
        """
        Single Adagrad parameter update with regularization.

        :param param_name: str - parameter name.
        :param param: Value - holds parameter state (data and gradient).
        """
        grad_squared = np.square(param.grad)
        grad_accumulated = self._accumulated.get(param_name, np.zeros_like(param.grad))
        self._accumulated[param_name] = grad_accumulated + grad_squared

        adaptive_lr = self.lr / np.sqrt(self._accumulated[param_name] + self.eps)

        param.data -= (adaptive_lr * param.grad) + (param.data * self.weight_decay)


class RMSProp(_Optim):
    """
    RMSProp gradient optimization algorithm.
    """

    def __init__(self, params: dict[str, Value], lr: float = 1e-2, weight_decay: float = 0.0, rho: float = 0.99,
                 eps: float = 1e-8):
        """
        :param params: dict (parameter_name, parameter) - contains all neural network parameters wrapped.
        into dictionary where key is parameter name in a specific layer and value is weight/bias with its own
        data and gradient of that specific layer.
        :param lr: float - learning rate of a neural network model.
        :param weight_decay: float - regularization parameter.
        :param rho: float - momentum factor.
        :param eps: float - value added to numerical stability in the denominator.
        """
        super().__init__(params=params, lr=lr, weight_decay=weight_decay)
        self.rho = rho
        self.eps = eps
        self._accumulated = {}

    def _update(self, param_name: str, param: Value):
        """
        Single RMRSProp parameter update with regularization.

        :param param_name: str - parameter name.
        :param param: Value - holds parameter state (data and gradient).
        """
        grad_squared = np.square(param.grad)
        grad_accumulated = self._accumulated.get(param_name, np.zeros_like(param.grad))
        self._accumulated[param_name] = self.rho * grad_accumulated + (1 - self.rho) * grad_squared

        adaptive_lr = self.lr / np.sqrt(self._accumulated[param_name] + self.eps)

        param.data -= (adaptive_lr * param.grad) + (param.data * self.weight_decay)


class Adam(_Optim):
    """
    Adam gradient optimization algorithm.
    """

    def __init__(self,
                 params: dict[str, Value],
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 lr: float = 1e-2,
                 weight_decay: float = 0.0,
                 eps: float = 1e-8,
                 t: int = 0
                 ):
        """
        :param params: dict (parameter_name, parameter) - contains all neural network parameters wrapped
        into dictionary where key is parameter name in a specific layer and value is weight/bias with its own
        data and gradient of that specific layer.
        :param beta1: float - coefficient used for computing running averages.
        :param beta2: float - coefficient used for computing running averages.
        :param lr: float - learning rate of a neural network model.
        :param weight_decay: float - regularization parameter.
        :param eps: float - value added to numerical stability in the denominator.
        :param t: int - iteration number for warming up Adam.
        """

        super().__init__(params=params, lr=lr, weight_decay=weight_decay)

        self._velocities = {}
        self._accumulated = {}
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = t

    def _update(self, param_name: str, param: Value):
        """
        Single Adam parameter update with regularization.

        :param param_name: str - parameter name.
        :param param: Value - holds parameter state (data and gradient).
        """
        self.t += 1

        velocity = self._velocities.get(param_name, np.zeros_like(param.data))
        self._velocities[param_name] = self.beta1 * velocity + (1 - self.beta1) * param.grad
        vt = self._velocities[param_name] / (1 - self.beta1 ** self.t)

        accumulated = self._accumulated.get(param_name, np.zeros_like(param.data))
        self._accumulated[param_name] = self.beta2 * accumulated + (1 - self.beta2) * param.grad ** 2
        at = self._accumulated[param_name] / (1 - self.beta2 ** self.t)

        param.data -= self.lr * vt / (np.sqrt(at) + self.eps) + (self.weight_decay * param.data)

