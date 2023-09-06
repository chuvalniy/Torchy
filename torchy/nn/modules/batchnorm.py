import numpy as np

from .module import Module
from torchy.nn.values import Value


class _BatchNorm(Module):
    """
    Base-like class for batch normalization
    """

    def __init__(self,
                 n_output: int,
                 eps: float = 1e-5,
                 momentum: float = 0.9):
        """
        :param n_output: int - number of output parameters.
        :param eps: float - values added to numerical stability in denominator.
        :param momentum: float - coefficient for computing running mean and variance
        """
        super(_BatchNorm, self).__init__()

        self._out = None
        self._X_norm = None
        self._X_var = None
        self._X_mean = None
        self.x = None

        self.eps = eps
        self.gamma = Value(np.ones(n_output))
        self.beta = Value(np.zeros(n_output))
        self.momentum = momentum

        self.running_mean = np.zeros(n_output)
        self.running_var = np.zeros(n_output)

    def _forward_1d(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of batch normalization layer.

        :param x: numpy array (batch_size, n_output) - incoming data.
        :return: numpy array (batch_size, n_output) - result of batchnorm processing.
        """
        if not self._train:
            eval_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self._out = self.gamma.data * eval_norm + self.beta.data
            return self._out

        self.x = np.copy(x)

        self._X_mean = np.mean(x, axis=0, keepdims=True)
        self._X_var = np.var(x, axis=0, keepdims=True)
        self._X_norm = (x - self._X_mean) / np.sqrt(self._X_var + self.eps)

        self._out = self.gamma.data * self._X_norm + self.beta.data

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self._X_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self._X_var

        return self._out

    def _backward_1d(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to x, gamma and beta.

        :param d_out: numpy array (batch_size, n_output) - gradient of loss function with respect to forward pass.
        :return: numpy array (batch_size, n_output) - gradient with respect to self.x.
        """
        self.gamma.grad = (self._X_norm * d_out).sum(axis=0)
        self.beta.grad = d_out.sum(0)

        batch_size = self.x.shape[0]

        sqrt_var_eps = np.sqrt(self._X_var + self.eps)
        dxhat = d_out * self.gamma.data
        dvar = np.sum(dxhat * (self.x - self._X_mean), axis=0) * (-1 / 2) * (self._X_var + self.eps) ** (-3 / 2)
        dmu = np.sum(dxhat * (-1 / sqrt_var_eps), axis=0) + dvar * (-2 / batch_size) * np.sum(self.x - self._X_mean,
                                                                                              axis=0)
        dx = dxhat * (1 / sqrt_var_eps) + dvar * (2 / batch_size) * (self.x - self._X_mean) + dmu / batch_size
        return dx


class BatchNorm1d(_BatchNorm):
    """
    Batch Normalization for one-dimensional layers.
    """

    def forward(self, x: np.ndarray, *args) -> np.ndarray:
        """
        Forward pass of BatchNorm1d layer.

        :param x: numpy array (batch_size, n_output) - incoming data.
        :return: numpy array (batch_size, n_output) - result of batchnorm processing.
        """
        return self._forward_1d(x)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to x, gamma and beta.

        :param d_out: numpy array (batch_size, n_output) - gradient of loss function with respect to forward pass.
        :return: numpy array (batch_size, n_output) - gradient with respect to self.x.
        """
        return self._backward_1d(d_out)


class BatchNorm2d(_BatchNorm):
    def forward(self, x: np.ndarray, *args) -> np.ndarray:
        """
        Forward pass of BatchNorm2d layer.

        Transforms input data to a two-dimensional array and basically calculates
        batch normalization in one-dimensional representation via _forward_ndim().

        After performing calculation, transforms output data to appropriate shape.

        :param x: numpy array (batch_size, in_channels, height, width) - incoming data.
        :return: numpy array (batch_size, in_channels, height, width) - result of batchnorm processing.
        """
        batch_size, in_channels, height, width = x.shape
        x_reshaped = x.transpose((0, 2, 3, 1)).reshape((-1, in_channels))
        out = self._forward_1d(x_reshaped)
        return out.reshape((batch_size, height, width, in_channels)).transpose((0, 3, 1, 2))

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to x, gamma and beta.

        In terms of shape, d_out acts the same as x in forward pass.

        :param d_out: numpy array (batch_size, n_output) - gradient of loss function with respect to forward pass.
        :return: numpy array (batch_size, n_output) - gradient with respect to self.x.
        """
        batch_size, in_channels, height, width = d_out.shape
        d_out_reshaped = d_out.transpose((0, 2, 3, 1)).reshape((-1, in_channels))
        dx = self._backward_1d(d_out_reshaped)
        return dx.reshape((batch_size, height, width, in_channels)).transpose((0, 3, 1, 2))
