import numpy as np

import module
from module import Module
from value import Value


def temp_softmax(x, y, mask, verbose=False):
    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print("dx_flat: ", dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx


def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.

    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


if __name__ == '__main__':
    class CustomRNN(Module):
        def __init__(self):
            super(CustomRNN, self).__init__()

            N, D, W, H = 10, 20, 30, 40
            word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
            V = len(word_to_idx)
            T = 13

            self.null = word_to_idx["<NULL>"]

            self.linear1 = module.Linear(D, H)

            self.embedding2 = module.Embedding(V, W)
            self.embedding2.weight = Value(np.random.randn(V, W) / 100)

            self.rnn3 = module.RNN(W, H)

            self.linear4 = module.Linear(H, V)

            self.features = np.linspace(-1.5, 0.3, num=(N * D)).reshape(N, D)
            self.captions = (np.arange(N * T) % V).reshape(N, T)

            self._init_params()

        def _init_params(self) -> None:
            for attr_name, attr_value in vars(self).items():
                if isinstance(attr_value, Module):
                    for param_name, param_value in attr_value.params.items():
                        attr_value.params[param_name] = np.linspace(-1.4, 1.3, num=param_value.data.size).reshape(
                            *param_value.data.shape)

        def forward(self, x: np.ndarray, *args) -> np.ndarray:
            h0 = self.linear1(self.features)
            emb = self.embedding2(self.captions[:, :-1])
            out, _ = self.rnn3(emb, h0)

            scores, _ = temporal_affine_forward(out, self.linear4.weight.data, self.linear4.bias.data.reshape(3,))

            return scores


    rnn = CustomRNN()
    out = rnn.forward(np.zeros(shape=(1,)))
    loss, _ = temp_softmax(out, rnn.captions[:, 1:], rnn.captions[:, 1:] != rnn.null)

    print(loss)
