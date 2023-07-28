import numpy as np


def _compute_softmax(y_pred):
    y_pred_copy = y_pred - np.max(y_pred, axis=1)

    return np.exp(y_pred_copy) / np.sum(np.exp(y_pred))


class CrossEntropyLoss:

    def __call__(self, y_pred, y_true):
        softmax = _compute_softmax(y_pred)

        loss = -np.sum(y_true * np.log(softmax), axis=1)
        grad = y_pred - softmax

        return loss, grad
