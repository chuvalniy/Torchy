import numpy as np


def compute_softmax(y_pred):
    exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


class CrossEntropyLoss:

    def __call__(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        num_classes = y_pred.shape[1]

        y_true_one_hot = np.zeros((batch_size, num_classes))
        y_true_one_hot[np.arange(batch_size), y_true] = 1

        logits = compute_softmax(y_pred)

        loss = -np.sum(y_true_one_hot * np.log(logits))
        grad = logits - y_true_one_hot

        return loss, grad
