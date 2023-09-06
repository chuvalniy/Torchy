import numpy as np

from .module import Module


class ReLU(Module):
    """
    Rectified Linear Unit activation function.
    """

    def __init__(self):
        super(ReLU, self).__init__()

        self._mask = None

    def forward(self, x: np.ndarray, *args) -> np.ndarray:
        """
        Forward pass of ReLU layer

        :param x: numpy array (n-dimensional) - incoming data.
        :return: numpy array (n-dimensional) - data after performing activation function on it, same shape
        as x.
        """
        self._mask = x > 0
        return np.where(self._mask, x, 0)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to incoming data.

        :param d_out: numpy array (n-dimensional) - gradient of loss function with respect to output of forward pass.
        :return: numpy array (n-dimensional) - gradient with respect to x, the same shape as d_out.
        """
        return d_out * self._mask


class Tanh(Module):
    """
    Hyperbolic tangent activation function.
    """

    def __init__(self):
        super(Tanh, self).__init__()
        self.x = None

    def forward(self, x: np.ndarray, *args) -> np.ndarray:
        """
        Forward pass of Tanh layer

        :param x: numpy array (n-dimensional) - incoming data.
        :return: numpy array (n-dimensional) - data after performing activation function on it, same shape
        as x.
        """
        self.x = np.tanh(x)
        return self.x

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to incoming data.

        :param d_out: numpy array (n-dimensional) - gradient of loss function with respect to output of forward pass.
        :return: numpy array (n-dimensional) - gradient with respect to x, the same shape as d_out.
        """
        return (1 - self.x ** 2) * d_out
