import numpy as np

from .module import Module
from ..initializations import init
from ..values import Value


class RNN(Module):
    """
    Vanilla recurrent neural network.
    """

    def __init__(self, n_input: int, n_output: int, bias: bool = True, nonlinearity: str = "tanh"):
        """
        :param n_input: int - size of each input sample.
        :param n_output: int - size of each output sample.
        :param bias: bool - consider bias in layer computation or not.
        :param nonlinearity: str - which nonlinearity to use in neural network.
        """
        super(RNN, self).__init__()
        self.dh = None
        self.n_input = n_input
        self.n_output = n_output

        self.weight_xh: Value = init.kaiming_uniform(shape=(n_input, n_output), nonlinearity=nonlinearity)
        self.weight_hh: Value = init.kaiming_uniform(shape=(n_output, n_output), nonlinearity=nonlinearity)
        self.bias: Value | None = init.kaiming_uniform(shape=(n_output,), nonlinearity=nonlinearity) if bias else None

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
