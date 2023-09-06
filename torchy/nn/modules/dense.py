import numpy as np

from torchy.nn.initializations import init
from torchy.nn.values import Value
from .module import Module


class Linear(Module):
    """
    Fully-connected / Dense layer
    """

    def __init__(self, n_input: int, n_output: int, bias: bool = True):
        """
        :param n_input: int - size of each input sample.
        :param n_output: int - size of each output sample.
        :param bias: bool - consider bias in layer computation or not
        """
        super(Linear, self).__init__()

        self.weight: Value = init.kaiming_uniform(shape=(n_input, n_output))
        self.bias: Value | None = init.kaiming_uniform(shape=(n_output,)) if bias else None
        self.X = None
        self.out = None

    def forward(self, x: np.ndarray, *args) -> np.ndarray:
        """
        Computes forward pass for linear layer.

        :param x: numpy array (batch_size, n_input) - incoming data.
        :return: numpy array (batch_size, n_output) - incoming data after linear transformation.
        """
        self.X = np.copy(x)
        self.out = np.dot(self.X, self.weight.data)

        if self.bias is not None:
            self.out += self.bias.data

        return self.out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes gradient with respect to input, weight and bias

        :param d_out: numpy array (batch_size, n_output) - gradient of loss with respect to an output.
        :return: numpy array (batch_size, n_input) - gradient with respect to input.
        """
        self.weight.grad = np.dot(self.X.T, d_out)

        if self.bias is not None:
            self.bias.grad = np.sum(d_out, axis=0).T

        d_pred = np.dot(d_out, self.weight.data.T)

        return d_pred
