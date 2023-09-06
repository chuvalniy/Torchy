from abc import ABC

import numpy as np

from torchy.nn.values import Value


class Module(ABC):
    """
    Abstract class that represents any layer behavior and also used for type-hinting.
    """

    def __init__(self):
        self._train = True
        self._params = {}

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes gradient with respect to input.

        :param d_out: numpy array (n-dimensional) - gradient of loss with respect to output.
        :return: numpy array (n-dimensional) - gradient with respect to input.
        """
        pass

    def forward(self, x: np.ndarray, *args) -> np.ndarray:
        """
        Interface method that computes forward pass for layer.

        :param x: numpy array (n-dimensional) - previous forward pass.
        :return: numpy array(n-dimensional) - forward pass for this layer.
        """
        pass

    def __call__(self, x: np.ndarray, *args) -> np.ndarray:
        return self.forward(x, *args)

    def eval(self):
        """
        Set layer to evaluation mode
        """
        self._train = False

    def train(self):
        """
        Set layer to training mode
        """
        self._train = True

    @property
    def params(self) -> dict[str, Value]:
        """
        Collects all module parameters into dictionary.

        :return: dict[str, Value] - layer parameters.
        """
        if self._params is None:
            return {}

        if len(self._params) == 0:
            self._init_params()
            if len(self._params) == 0:
                self._params = None
                return {}

        return self._params

    def _init_params(self):
        """
        Goes through module attributes and sets it as parameter if it is an instance of Value.

        :return: None
        """
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, Value):
                self._params[attr_name] = attr_value
