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

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> (float, np.ndarray):
        """

        :param y_pred: numpy array (batch_size, n_output) - predictions computed by neural network with a range
        of values from (-infinity, +infinity).
        :param y_true: numpy array (batch_size) - indices of ground truth values.
        :return: loss (float) - cross-entropy loss.
        :return grad (batch_size, n_output) - gradient of loss function with respect to softmax.
        """
        batch_size = y_true.shape[0]
        num_classes = y_pred.shape[1]

        y_true_one_hot = np.identity(num_classes)[y_true.reshape(-1)]
        logits = compute_softmax(y_pred)

        loss = -np.sum(y_true_one_hot * np.log(logits + 1e-10)) / batch_size

        grad = (logits - y_true_one_hot) / batch_size

        return loss, grad
