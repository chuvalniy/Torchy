import numpy as np

from .module import Module
from torchy.nn.values import Value
from torchy.nn.initializations import init


class Conv2d(Module):
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

        self.weight: Value = init.kaiming_uniform(shape=(out_channels, in_channels, kernel_size, kernel_size))
        self.bias: Value | None = init.kaiming_uniform(shape=(out_channels,)) if bias else None
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
