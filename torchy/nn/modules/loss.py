from typing import Tuple

import numpy as np


def compute_softmax(y_pred: np.ndarray) -> np.ndarray:
    """
    Computes softmax function and subtracts the maximum values from each row for calculation stability.

    :param y_pred: numpy array (batch_size, n_output) - predictions computed by neural network with a range
    of values from (-infinity, +infinity).
    :return: numpy array (batch_size, n_output) - result of softmax function with a range of values from (0, 1)
    with a total row sum of 1.
    """
    exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


class CrossEntropyLoss:
    """
    Cross-entropy Loss

    Computes the cross-entropy loss between input logits and the target
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
        """Forward and backward pass for Cross Entropy Loss.

        :param y_pred: numpy array (batch_size, n_output) - predictions computed by neural network with a range
        of values from (-infinity, +infinity).
        :param y_true: numpy array (batch_size) - indices of ground truth values.
        :return: loss (float) - KL divergence loss with its gradients.
        """
        batch_size = y_true.shape[0]
        num_classes = y_pred.shape[1]

        y_true_one_hot = np.identity(num_classes)[y_true.reshape(-1)]
        logits = compute_softmax(y_pred)

        loss = -np.sum(y_true_one_hot * np.log(logits + 1e-10)) / batch_size

        grad = (logits - y_true_one_hot) / batch_size

        return loss, grad


class KLDivLoss:
    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
        """Forward and backward pass for KL Divergence.

        :param y_pred: numpy array - predictions computed by neural network with a range
        :param y_true: numpy array - target distribution
        :return: loss (float) - KL divergence loss with its gradients.
        """
        assert y_true.shape == y_pred.shape, "Invalid shape for y_pred and y_true"

        batch_size = y_true.shape[0]

        loss = np.sum(
            y_true * np.log((y_true + self.eps) / (y_pred + self.eps))
        )
        grad = -y_true / (y_pred + self.eps)

        return loss / batch_size, grad / batch_size
