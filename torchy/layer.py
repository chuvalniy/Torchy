import copy
from abc import ABC

import numpy as np

from tools import kaiming_init
from torchy.value import Value


class Layer(ABC):
    """
    Abstract class that represents any layer behavior and also used for type-hinting.
    """

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes gradient with respect to input.

        :param d_out: numpy array (n-dimensional) - gradient of loss with respect to output.
        :return: numpy array (n-dimensional) - gradient with respect to input.
        """
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Interface method that computes forward pass for layer.

        :param x: numpy array (n-dimensional) - previous forward pass.
        :return: numpy array(n-dimensional) - forward pass for this layer.
        """
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class NnLayer(Layer):
    """
    Another abstract class to represent layers that can hold parameters (weight & bias).
    """

    def zero_grad(self):
        """
        Goes through all the parameters in layer and sets their gradient to zero.

        :return: None
        """
        pass

    def params(self) -> dict[str, Value]:
        """
        Collects all layer parameters into dictionary.

        :return: dict[str, Value] - layer parameters.
        """

        pass


class Linear(NnLayer):
    """
    Fully-connected / Dense layer
    """

    def __init__(self, n_input: int, n_output: int, bias: bool = True):
        """
        :param n_input: int - size of each input sample.
        :param n_output: int - size of each output sample.
        :param bias: bool - consider bias in layer computation or not
        """
        self.W = Value(kaiming_init(n_input) * np.random.randn(n_input, n_output))
        self.B = Value(kaiming_init(n_input) * np.random.randn(1, n_output)) if bias else None
        self.X = None
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes forward pass for linear layer.

        :param x: numpy array (batch_size, n_input) - incoming data.
        :return: numpy array (batch_size, n_output) - incoming data after linear transformation.
        """
        self.X = copy.deepcopy(x)
        self.out = np.dot(self.X, self.W.data)

        if self.B is not None:
            self.out += self.B.data

        return self.out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes gradient with respect to input, weight and bias

        :param d_out: numpy array (batch_size, n_output) - gradient of loss with respect to an output.
        :return: numpy array (batch_size, n_input) - gradient with respect to input.
        """
        self.W.grad = np.dot(self.X.T, d_out)

        if self.B is not None:
            self.B.grad = np.sum(d_out, axis=0).T

        d_pred = np.dot(d_out, self.W.data.T)

        return d_pred

    def zero_grad(self):
        """
        Goes through layer parameters and zeroes their gradient

        :return: None
        """
        for _, param in self.params().items():
            param.grad = np.zeros_like(param.grad)

    def params(self) -> dict[str, Value]:
        """
        Collects all layer parameters into dictionary.

        :return: dict[str, Value] - layer parameters.
        """

        d = {
            "W": self.W,
        }

        if self.B is not None:
            d["B"] = self.B

        return d


class Conv2d(NnLayer):
    """
    Two-dimensional convolutional neural network layer
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True
    ):
        """
        :param in_channels: int - number of input channels.
        :param out_channels: int - number of output channels after convolution.
        :param kernel_size: int - size of convolutional kernel.
        :param stride: int - stride of convolutional kernel (default = 1).
        :param padding: int - padding added to all axes with respect to input (default = 0).
        :param bias: bool - consider bias in layer computation or not
        """
        self.W = Value(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.B = Value(np.zeros(out_channels)) if bias else None
        self.X = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self._padding_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes forward pass for convolutional layer.

        :param x: numpy array (batch_size, in_channels, height, width) - incoming data.
        :return: numpy array (batch_size, out_channels, out_height, out_width) - incoming data after
        performing convolution operation on it.
        """
        self.X = np.pad(x, pad_width=self._padding_width, mode="constant", constant_values=0)
        batch_size, in_channels, height, width = x.shape

        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        W_flatten = np.reshape(self.W.data, (-1, self.out_channels))

        out = np.zeros(shape=(batch_size, self.out_channels, out_height, out_width))
        for y in range(0, out_height, self.stride):
            for x in range(0, out_width, self.stride):
                input_region = self.X[:, :, y:y + self.kernel_size, x:x + self.kernel_size]
                input_region_flatten = np.reshape(input_region, (batch_size, -1))

                output_feature_map = np.dot(input_region_flatten, W_flatten)
                if self.B is not None:
                    output_feature_map += self.B.data

                out[:, :, y, x] = output_feature_map

        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes gradient for convolutional layer with respect to input, weight and bias and also removes padding
        from the input layer that was added in the forward pass.

        :param d_out: numpy array (batch_size, out_channels, out_height, out_width) - gradient of loss function with
        respect to output of forward pass.
        :return: numpy array (batch_size, in_channels, height, width) - gradient with respect to input.
        """
        batch_size, out_channels, out_height, out_width = d_out.shape

        W_flatten = self.W.data.reshape(-1, self.out_channels)
        W_flatten_grad = self.W.grad.reshape(-1, self.out_channels)

        d_pred = np.zeros_like(self.X)
        for y in range(0, out_height, self.stride):
            for x in range(0, out_width, self.stride):
                output_region = self.X[:, :, y:y + self.kernel_size, x:x + self.kernel_size]
                output_region_flatten = np.reshape(output_region, (batch_size, -1))
                pixel = d_out[:, :, y, x]
                d_output_region_flatten = np.dot(pixel, W_flatten.T)
                d_output_region = np.reshape(d_output_region_flatten, output_region.shape)
                d_pred[:, :, y:y + self.kernel_size, x:x + self.kernel_size] += d_output_region

                W_flatten_grad += np.dot(output_region_flatten.T, pixel)

        if self.B is not None:
            self.B.grad = np.sum(d_out, axis=(0, 2, 3))

        return d_pred[:, :, 1:-1, 1:-1]

    def params(self) -> dict[str, Value]:
        """
        Collects all layer parameters into dictionary.

        :return: dict[str, Value] - layer parameters.
        """
        d = {
            "W": self.W,
        }

        if self.B is not None:
            d["B"] = self.B

        return d


class ReLU(Layer):
    """
    Rectified Linear Unit activation function.
    """

    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
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


class BatchNorm1d(NnLayer):
    """
    Batch Normalization for one-dimensional layers.
    """

    def __init__(self,
                 n_output: int,
                 eps: float = 1e-5,
                 momentum: float = 0.9):
        """
        :param n_output: int - number of output parameters.
        :param eps: float - value added to numerical stability in denominator.
        :param momentum: float - coefficient for computing running mean and variance
        """
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of BatchNorm1d layer.

        :param x: numpy array (batch_size, n_output) - incoming data.
        :return: numpy array (batch_size, n_output) - result of batchnorm processing.
        """
        self.x = copy.deepcopy(x)

        self._X_mean = np.mean(x, axis=0, keepdims=True)
        self._X_var = np.var(x, axis=0, keepdims=True)
        self._X_norm = (x - self._X_mean) / np.sqrt(self._X_var + self.eps)

        self._out = self.gamma.data * self._X_norm + self.beta.data

        self.running_mean = self.momentum * self.running_var + (1 - self.momentum) * self._X_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self._X_var

        return self._out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to self.x, self.gamma and self.beta.

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

    def params(self) -> dict[str, Value]:
        """
        Collects all layer parameters into dictionary.

        :return: dict[str, Value] - layer parameters.
        """
        d = {
            "gamma": self.gamma,
            "beta": self.beta
        }

        return d


class MaxPool2d(Layer):
    """
    Max pooling layer for 2-dimensional input (Convolutional Layer).
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        :param kernel_size: int - size of max pooling kernel.
        :param stride: int - stride of max pooling  kernel (default = kernel_size).
        :param padding: int = padding added to all axis with respect to input (default = 0).
        """
        self.X = None

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self._padding_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of max pooling layer.

        :param x: numpy array (batch_size, in_channels, height, width) - data to perform max pooling.
        :return: numpy array (batch_size, in_channels, out_height, out_width) - result of max pooling 'x'.
        """
        self.X = np.pad(x, pad_width=self._padding_width, mode="constant", constant_values=0)
        batch_size, in_channels, height, width = self.X.shape

        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        out = np.zeros(shape=(batch_size, in_channels, out_height, out_width))
        for y in range(0, out_height, self.stride):
            for x in range(0, out_width, self.stride):
                input_region = self.X[:, :, y:self.kernel_size + y, x:self.kernel_size + x]
                out[:, :, y, x] += np.max(input_region, axis=(2, 3))

        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to self.x.

        :param d_out: numpy array (batch_size, in_channels, out_height, out_width) - gradient of loss function with
        respect to output of forward pass.
        :return: numpy array (batch_size, in_channels, height, width) - gradient with respect to self.x.
        """
        _, in_channels, out_height, out_width = d_out.shape

        d_pred = np.zeros_like(self.X)
        for y in range(0, out_height, self.stride):
            for x in range(0, out_width, self.stride):
                output_region = self.X[:, :, y:y + self.kernel_size, x:x + self.kernel_size]
                grad = d_out[:, :, y, x][:, :, np.newaxis, np.newaxis]
                mask = (output_region == np.max(output_region, (2, 3))[:, :, np.newaxis, np.newaxis])
                d_pred[:, :, y:y + self.kernel_size, x:x + self.kernel_size] += grad * mask

        return d_pred
