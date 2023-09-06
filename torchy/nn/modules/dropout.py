import numpy as np

from .module import Module


class Dropout(Module):
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
