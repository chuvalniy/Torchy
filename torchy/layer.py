import copy
from abc import ABC

import numpy as np

from tools import kaiming_init
from torchy.value import Value


class Layer(ABC):
    """
    Abstract class that represents any layer behavior and also used for type-hinting.
    """

    def __init__(self):
        self._train = True

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


class NnLayer(Layer):
    """
    Another abstract class to represent layers that can hold parameters (weight & bias).
    """

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
        super(Linear, self).__init__()

        self.W = Value(kaiming_init(n_input) * np.random.randn(n_input, n_output))
        self.B = Value(np.zeros(shape=(1, n_output))) if bias else None
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
        super(Conv2d, self).__init__()

        init_value = kaiming_init(in_channels * kernel_size * kernel_size)
        self.weight = Value(init_value * np.random.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = Value(np.zeros(out_channels)) if bias else None
        self.x = None

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
        self.x = x
        x_padded = np.pad(x, pad_width=self._padding_width)

        batch_size, in_channels, height, width = x.shape

        height += 2 * self.padding
        width += 2 * self.padding

        out_height = 1 + (height - self.kernel_size) // self.stride
        out_width = 1 + (width - self.kernel_size) // self.stride

        w_flattened = np.transpose(self.weight.data, axes=(2, 3, 1, 0)).reshape((-1, self.out_channels))

        out = np.zeros(shape=(batch_size, self.out_channels, out_height, out_width))
        for oh in range(out_height):
            for ow in range(out_width):
                oh_step = self.stride * oh
                ow_step = self.stride * ow

                input_region = x_padded[:, :, oh_step:oh_step + self.kernel_size, ow_step:ow_step + self.kernel_size]
                input_region_flattened = input_region.transpose((0, 2, 3, 1)).reshape((batch_size, -1))

                out[:, :, oh, ow] = np.dot(input_region_flattened, w_flattened)
                if self.bias is not None:
                    out[:, :, oh, ow] += self.bias.data

        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes gradient for convolutional layer with respect to input, weight and bias and also removes padding
        from the input layer that was added in the forward pass.

        :param d_out: numpy array (batch_size, out_channels, out_height, out_width) - gradient of loss function with
        respect to output of forward pass.
        :return: numpy array (batch_size, in_channels, height, width) - gradient with respect to input.
        """
        batch_size, in_channels, height, width = self.x.shape
        _, out_channels, out_height, out_width = d_out.shape

        x_padded = np.pad(self.x, pad_width=self._padding_width)

        w_flattened = self.weight.data.transpose((2, 3, 1, 0)).reshape((-1, self.out_channels))

        self.weight.grad = np.zeros(shape=(self.kernel_size, self.kernel_size, in_channels, out_channels))
        dw_flattened = self.weight.grad.reshape((-1, out_channels))

        d_x = np.zeros_like(x_padded)
        for oh in range(out_height):
            for ow in range(out_width):
                oh_step = self.stride * oh
                ow_step = self.stride * ow

                input_region = x_padded[:, :, oh_step:oh_step + self.kernel_size, ow_step:ow_step + self.kernel_size]
                input_region_flattened = input_region.transpose((0, 2, 3, 1)).reshape((batch_size, -1))
                d_out_pixel = d_out[:, :, oh, ow]

                dw_flattened += np.dot(input_region_flattened.T, d_out_pixel)
                dx_region_flattened = np.dot(d_out_pixel, w_flattened.T)
                dx_region = np.transpose(
                    dx_region_flattened.reshape((batch_size, self.kernel_size, self.kernel_size, in_channels)),
                    axes=(0, 3, 1, 2)
                )

                d_x[:, :, oh_step:oh_step + self.kernel_size, ow_step: ow_step + self.kernel_size] += dx_region

        if self.bias is not None:
            self.bias.grad = np.sum(d_out, axis=(0, 2, 3))

        self.weight.grad = self.weight.grad.transpose((3, 2, 0, 1))

        return d_x[:, :, self.padding:height + self.padding, self.padding:width + self.padding]

    def params(self) -> dict[str, Value]:
        """
        Collects all layer parameters into dictionary.

        :return: dict[str, Value] - layer parameters.
        """
        d = {
            "W": self.weight,
        }

        if self.bias is not None:
            d["B"] = self.bias

        return d


class ReLU(Layer):
    """
    Rectified Linear Unit activation function.
    """

    def __init__(self):
        super(ReLU, self).__init__()

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
        super(BatchNorm1d, self).__init__()

        self._out = None
        self._X_norm = None
        self._X_var = None
        self._X_mean = None
        self._X = None

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
        if not self._train:
            eval_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self._out = self.gamma.data * eval_norm + self.beta.data
            return self._out

        self._X = copy.deepcopy(x)

        self._X_mean = np.mean(x, axis=0, keepdims=True)
        self._X_var = np.var(x, axis=0, keepdims=True)
        self._X_norm = (x - self._X_mean) / np.sqrt(self._X_var + self.eps)

        self._out = self.gamma.data * self._X_norm + self.beta.data

        self.running_mean = self.momentum * self.running_var + (1 - self.momentum) * self._X_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self._X_var

        return self._out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to x, gamma and beta.

        :param d_out: numpy array (batch_size, n_output) - gradient of loss function with respect to forward pass.
        :return: numpy array (batch_size, n_output) - gradient with respect to self.x.
        """
        self.gamma.grad = (self._X_norm * d_out).sum(axis=0)
        self.beta.grad = d_out.sum(0)

        batch_size = self._X.shape[0]

        sqrt_var_eps = np.sqrt(self._X_var + self.eps)
        dxhat = d_out * self.gamma.data
        dvar = np.sum(dxhat * (self._X - self._X_mean), axis=0) * (-1 / 2) * (self._X_var + self.eps) ** (-3 / 2)
        dmu = np.sum(dxhat * (-1 / sqrt_var_eps), axis=0) + dvar * (-2 / batch_size) * np.sum(self._X - self._X_mean,
                                                                                              axis=0)
        dx = dxhat * (1 / sqrt_var_eps) + dvar * (2 / batch_size) * (self._X - self._X_mean) + dmu / batch_size
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
        super(MaxPool2d, self).__init__()

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
        for oh in range(out_height):
            for ow in range(out_width):
                oh_step = self.stride * oh
                ow_step = self.stride * ow

                input_region = self.X[:, :, oh_step:self.kernel_size + oh_step, ow_step:self.kernel_size + ow_step]
                out[:, :, oh, ow] += np.max(input_region, axis=(2, 3))

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
        for oh in range(out_height):
            for ow in range(out_width):
                oh_step = self.stride * oh
                ow_step = self.stride * ow

                output_region = self.X[:, :, oh_step:oh_step + self.kernel_size, ow_step:ow_step + self.kernel_size]
                grad = d_out[:, :, oh, ow][:, :, np.newaxis, np.newaxis]
                mask = (output_region == np.max(output_region, (2, 3))[:, :, np.newaxis, np.newaxis])
                d_pred[:, :, oh_step:oh_step + self.kernel_size, ow_step:ow_step + self.kernel_size] += grad * mask

        return d_pred


class Dropout(Layer):
    """
    Inverted dropout
    """

    def __init__(self, p: float = 0.5):
        """
        :param p: float - probability of keeping each neuron
        """
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs forward pass for inverted dropout.

        :param x: numpy array (n-dimensional) - incoming data.
        :return: numpy array (n-dimensional) - output data, same shape as incoming data
        """
        out = x.copy()
        if self._train:
            self.mask = (np.random.rand(*x.shape) < self.p) / self.p
            out *= self.mask

        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to forward pass input.

        :param d_out: numpy array (n-dimensional) - gradient of loss with respect to output of forward pass
        :return: numpy array (n-dinemsional) - gradient with respect to input, same shape as forward pass input
        """

        return d_out * self.mask
