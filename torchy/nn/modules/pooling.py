import numpy as np

from .module import Module


class MaxPool2d(Module):
    """
    Max pooling layer for 2-dimensional feature map (Convolutional Layer).
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
