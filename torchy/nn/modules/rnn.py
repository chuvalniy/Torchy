import numpy as np

from .activation import sigmoid
from .module import Module
from ..initializations import init
from ..values import Value

__all__ = ["RNN", "LSTM"]


class _BaseRNN(Module):
    def __init__(self, n_input: int, n_output: int, bias: bool = True, nonlinearity: str = "tanh"):
        """
        :param n_input: int - size of each input sample.
        :param n_output: int - size of each output sample.
        :param bias: bool - consider bias in layer computation or not.
        :param nonlinearity: str - which non-linearity to use (currently not supported).
        """
        super(_BaseRNN, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.weight_xh: Value = init.kaiming_uniform(shape=(n_input, n_output), nonlinearity=nonlinearity)
        self.weight_hh: Value = init.kaiming_uniform(shape=(n_output, n_output), nonlinearity=nonlinearity)
        self.bias: Value | None = init.kaiming_uniform(shape=(n_output,), nonlinearity=nonlinearity) if bias else None

        self.hidden_states = None
        self.x = None


class RNN(_BaseRNN):
    """
    Vanilla recurrent neural network.
    """

    def forward(self, x: np.ndarray, h0: np.ndarray = None) -> (np.ndarray, np.ndarray):
        """
        Forward pass for RNN in a 'for' loop way.

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
            h_raw = np.dot(self.x[:, idx, :], self.weight_xh.data) + np.dot(h, self.weight_hh.data)

            if self.bias is not None:
                h_raw += self.bias.data

            self.hidden_states[:, idx + 1, :] = np.tanh(h_raw)
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

        dh_next = np.zeros_like(self.hidden_states[:, 0, :])
        for idx in reversed(range(sequence_length)):
            dh_curr = d_out[:, idx, :] + dh_next
            dh_raw = (1 - np.square(self.hidden_states[:, idx + 1, :])) * dh_curr

            if self.bias is not None:
                self.bias.grad += dh_raw.sum(axis=0)
            self.weight_xh.grad += np.dot(self.x[:, idx, :].T, dh_raw)

            self.weight_hh.grad += np.dot(self.hidden_states[:, idx, :].T, dh_raw)
            dx[:, idx, :] = np.dot(dh_raw, self.weight_xh.data.T)

            dh_next = np.dot(dh_raw, self.weight_hh.data.T)

        return dx


class LSTM(_BaseRNN):
    """
    Vanilla LSTM neural network.
    """
    def __init__(self, n_input: int, n_output: int, bias: bool = True, nonlinearity: str = "tanh"):
        """
        :param n_input: int - size of each input sample.
        :param n_output: int - size of each output sample.
        :param bias: bool - consider bias in layer computation or not.
        :param nonlinearity: str - which non-linearity to use (currently not supported).
        """
        super(LSTM, self).__init__(n_input, n_output, bias, nonlinearity)

        self.cell_states = None

        self.weight_xh: Value = init.kaiming_uniform(shape=(n_input, 4 * n_output), nonlinearity=nonlinearity)
        self.weight_hh: Value = init.kaiming_uniform(shape=(n_output, 4 * n_output), nonlinearity=nonlinearity)
        self.bias = init.kaiming_uniform(shape=(4 * n_output,), nonlinearity=nonlinearity) if bias else None

        self.block_input = None
        self.output_gates = None
        self.forget_gates = None
        self.input_gates = None

    def forward(self, x: np.ndarray, h0: np.ndarray = None, c0: np.ndarray = None) -> (np.ndarray, np.ndarray):
        """
        Forward pass for LSTM neural network in a 'for' loop way.

        :param x: numpy array (batch_size, sequence_length, input_size) - incoming data.
        :param h0: numpy array (bach_size, hidden_size) - initial hidden state for the input sequence.
        :param c0:  numpy array (batch_size, hidden_size) - initial cell state for the input sequence.

        :return: tuple of two numpy arrays (output, hn):
                output: numpy array (batch_size, sequence_length, hidden_size) - hidden states for all time steps.
                hn: numpy array (batch_size, hidden_size) - final hidden state.
        """
        self.x = x.copy()
        batch_size, sequence_length, _ = x.shape

        h = np.copy(h0) if h0 is not None else np.zeros(shape=(batch_size, self.n_output))
        c = np.copy(c0) if c0 is not None else np.zeros(shape=(batch_size, self.n_output))

        self.hidden_states = np.zeros(shape=(batch_size, sequence_length + 1, self.n_output))
        self.hidden_states[:, 0, :] = h

        self.cell_states = np.zeros(shape=(batch_size, sequence_length + 1, self.n_output))
        self.cell_states[:, 0, :] = c

        self.input_gates = np.zeros(shape=(batch_size, sequence_length, self.n_output))
        self.forget_gates = np.zeros(shape=(batch_size, sequence_length, self.n_output))
        self.output_gates = np.zeros(shape=(batch_size, sequence_length, self.n_output))
        self.block_input = np.zeros(shape=(batch_size, sequence_length, self.n_output))

        for idx in range(sequence_length):
            a_raw = np.dot(h, self.weight_hh.data) + np.dot(self.x[:, idx, :], self.weight_xh.data)
            if self.bias is not None:
                a_raw += self.bias.data

            ai, af, ao, ag = np.array_split(a_raw, 4, axis=1)

            self.input_gates[:, idx, :] = sigmoid(ai)
            self.forget_gates[:, idx, :] = sigmoid(af)
            self.output_gates[:, idx, :] = sigmoid(ao)
            self.block_input[:, idx, :] = np.tanh(ag)

            c = self.forget_gates[:, idx, :] * c + self.input_gates[:, idx, :] * self.block_input[:, idx, :]
            h = self.output_gates[:, idx, :] * np.tanh(c)

            self.cell_states[:, idx + 1, :] = c
            self.hidden_states[:, idx + 1, :] = h

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

        dh_next = np.zeros_like(self.hidden_states[:, 0, :])
        dc_next = np.zeros_like(self.cell_states[:, 0, :])

        for idx in reversed(range(sequence_length)):
            dh_curr = d_out[:, idx, :] + dh_next
            dc_curr = self.output_gates[:, idx, :] * np.cosh(self.cell_states[:, idx + 1, :])**(-2) * dh_curr + dc_next

            doutput_gate = np.tanh(self.cell_states[:, idx + 1, :]) * dh_curr
            dforget_gate = self.cell_states[:, idx, :] * dc_curr

            dc_next = self.forget_gates[:, idx, :] * dc_curr

            dinput_gate = self.block_input[:, idx, :] * dc_curr
            dblock_input = self.input_gates[:, idx, :] * dc_curr

            d_ag = (1 - self.block_input[:, idx, :] ** 2) * dblock_input
            d_ao = (self.output_gates[:, idx, :] * (1 - self.output_gates[:, idx, :])) * doutput_gate
            d_af = (self.forget_gates[:, idx, :] * (1 - self.forget_gates[:, idx, :])) * dforget_gate
            d_ai = (self.input_gates[:, idx, :] * (1 - self.input_gates[:, idx, :])) * dinput_gate

            da = np.concatenate((d_ai, d_af, d_ao, d_ag), axis=1)

            self.weight_xh.grad += np.dot(self.x[:, idx, :].transpose(), da)
            self.weight_hh.grad += np.dot(self.hidden_states[:, idx, :].transpose(), da)
            if self.bias is not None:
                self.bias.grad += da.sum(axis=0)

            dx[:, idx, :] += np.dot(da, self.weight_xh.data.transpose())
            dh_prev = np.dot(da, self.weight_hh.data.transpose())

            dh_next = dh_prev

        return dx
