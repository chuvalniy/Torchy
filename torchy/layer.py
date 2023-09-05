import copy
from abc import ABC

import numpy as np

from torchy.tools import kaiming_init
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

        self.weight = Value(kaiming_init(n_input) * np.random.randn(n_input, n_output))
        self.bias = Value(np.zeros(shape=(1, n_output))) if bias else None
        self.X = None
        self.out = None

    def forward(self, x: np.ndarray, *args) -> np.ndarray:
        """
        Computes forward pass for linear layer.

        :param x: numpy array (batch_size, n_input) - incoming data.
        :return: numpy array (batch_size, n_output) - incoming data after linear transformation.
        """
        self.X = copy.deepcopy(x)
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

    def forward(self, x: np.ndarray, *args) -> np.ndarray:
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


class RNN(NnLayer):
    """
    Vanilla recurrent neural network.
    """

    def __init__(self, n_input: int, n_output: int, bias: bool = True):
        """
        :param n_input: int - size of each input sample.
        :param n_output: int - size of each output sample.
        :param bias: bool - consider bias in layer computation or not
        """
        super(RNN, self).__init__()
        self.dh = None
        self.n_input = n_input
        self.n_output = n_output

        self.weight_xh = Value(kaiming_init(n_output) * np.random.randn(n_input, n_output))
        self.weight_hh = Value(kaiming_init(n_output) * np.random.randn(n_output, n_output))
        self.bias = Value(np.zeros(n_output)) if bias else None

        self.hidden_states = None
        self.x = None

    def forward(self, x: np.ndarray, h0: np.ndarray = None) -> (np.ndarray, np.ndarray):
        """
        :param x: numpy array (batch_size, sequence_length, input_size) - incoming data.
        :param h0: numpy array (bach_size, hidden_size) - initial hidden state for the input sequence.
        :return: tuple of two numpy arrays (output, hn):
                output: numpy array (batch_size, sequence_length, hidden_size) - hidden states for all time steps.
                hn: numpy array (batch_size, hidden_size) - final hidden state.
        """
        self.x = x.copy()
        batch_size, sequence_length, _ = x.shape

        h = np.copy(h0) if h0 is not None else np.zeros(shape=(batch_size, self.n_output))
        self.hidden_states = np.zeros(shape=(batch_size, sequence_length + 1, self.n_output))
        self.hidden_states[:, 0, :] = h

        for idx in range(sequence_length):
            if self.bias is not None:
                self.hidden_states[:, idx + 1, :] = np.tanh(
                    np.dot(self.x[:, idx, :], self.weight_xh.data) + np.dot(h, self.weight_hh.data) + self.bias.data
                )
            else:
                self.hidden_states[:, idx + 1, :] = np.tanh(
                    np.dot(self.x[:, idx, :], self.weight_xh.data) + np.dot(h, self.weight_hh.data)
                )
            h = self.hidden_states[:, idx + 1, :]

        return self.hidden_states[:, 1:, :], self.hidden_states[:, -1, :]

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to incoming data.

        :param d_out: numpy array (batch_size, sequence_length, hidden_size) -
        gradient of loss function with respect to output of forward pass.

        :return: numpy array (batch_size, sequence_length, input_size) -
        gradient with respect to x, the same shape as d_out.
        """
        dx = np.zeros_like(self.x)
        _, sequence_length, _ = dx.shape

        dh_prev = np.zeros_like(self.hidden_states[:, 0, :])
        for idx in reversed(range(sequence_length)):
            dh_curr = d_out[:, idx, :] + dh_prev
            dh_raw = (1 - np.square(self.hidden_states[:, idx + 1, :])) * dh_curr

            if self.bias is not None:
                self.bias.grad += dh_raw.sum(axis=0)
            self.weight_xh.grad += np.dot(self.x[:, idx, :].T, dh_raw)

            self.weight_hh.grad += np.dot(self.hidden_states[:, idx, :].T, dh_raw)
            dx[:, idx, :] = np.dot(dh_raw, self.weight_xh.data.T)

            dh_prev = np.dot(dh_raw, self.weight_hh.data.T)

        return dx

    def params(self) -> dict[str, Value]:
        """
        Collects all layer parameters into dictionary.

        :return: dict[str, Value] - layer parameters.
        """
        d = {
            "WH": self.weight_hh,
            "WX": self.weight_xh
        }

        if self.bias is not None:
            d["B"] = self.bias

        return d


class Embedding(NnLayer):
    """
    Word embedding layer.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        :param num_embeddings: int - dictionary size.
        :param embedding_dim: int - embedding dimension for each word in dictionary.
        """
        super(Embedding, self).__init__()

        self.weight = Value(np.random.randn(num_embeddings, embedding_dim))
        self.x = None

    # TODO: docstring
    def forward(self, x: np.ndarray, *args) -> np.ndarray:
        """
        Computes forward pass for embedding layer.

        :param x: numpy array (batch_size, in_channels, height, width) - incoming data.
        :return: numpy array (batch_size, embedding_dim) - word embeddings
        performing convolution operation on it.
        """
        self.x = np.copy(x)

        return self.weight.data[self.x]

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        np.add.at(self.weight.grad, self.x, d_out)
        return self.weight.grad

    @classmethod
    def from_pretrained(cls, weight: np.ndarray) -> 'Embedding':
        num_embeddings, embedding_dim = weight.shape
        embedding = cls(num_embeddings, embedding_dim)
        embedding.weight = Value(weight)

        return embedding

    def params(self) -> dict[str, Value]:
        d = {
            "W": self.weight
        }

        return d


class ReLU(Layer):
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


class Tanh(Layer):
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


class _BatchNorm(NnLayer):
    """
    Base-like class for batch normalization
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
        super(_BatchNorm, self).__init__()

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

    def _forward_ndim(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of batch normalization layer.

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

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self._X_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self._X_var

        return self._out

    def _backward_ndim(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to self.x, self.gamma and self.beta.

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
        return self._forward_ndim(x)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to self.x, self.gamma and self.beta.

        :param d_out: numpy array (batch_size, n_output) - gradient of loss function with respect to forward pass.
        :return: numpy array (batch_size, n_output) - gradient with respect to self.x.
        """
        return self._backward_ndim(d_out)


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
        out = self._forward_ndim(x_reshaped)
        return out.reshape((batch_size, height, width, in_channels)).transpose((0, 3, 1, 2))

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Computes backward pass with respect to self.x, self.gamma and self.beta.

        In terms of shape, d_out acts the same as x in forward pass.

        :param d_out: numpy array (batch_size, n_output) - gradient of loss function with respect to forward pass.
        :return: numpy array (batch_size, n_output) - gradient with respect to self.x.
        """
        batch_size, in_channels, height, width = d_out.shape
        d_out_reshaped = d_out.transpose((0, 2, 3, 1)).reshape((-1, in_channels))
        dx = self._backward_ndim(d_out_reshaped)
        return dx.reshape((batch_size, height, width, in_channels)).transpose((0, 3, 1, 2))


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

    def forward(self, x: np.ndarray, *args) -> np.ndarray:
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

    def __init__(self, p: float = 0.5, seed: int = None):
        """
        :param p: float - probability of keeping each neuron.
        :param seed: int - parameter for data representativity (default None).
        """
        super(Dropout, self).__init__()
        self.p = p
        self.seed = np.random.seed(seed) if seed else None

        self.mask = None

    def forward(self, x: np.ndarray, *args) -> np.ndarray:
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
