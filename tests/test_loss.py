import numpy as np

from loss import CrossEntropyLoss, compute_softmax
from tests.gradient_check import GradientCheck


def test_softmax():
    probs = compute_softmax(np.array([[20, 0, 0], [1000, 0, 0]]))
    assert np.all(np.isclose(probs[:, 0], 1.0))


def test_cross_entropy_single():
    num_classes = 4
    batch_size = 1

    predictions = np.random.randint(-1, 3, size=(batch_size, num_classes)).astype(np.float64)
    y_true = np.random.randint(0, num_classes, size=(batch_size, 1)).astype(np.int32)

    criterion = CrossEntropyLoss()

    GradientCheck.check_gradient(lambda y_pred: criterion(y_pred, y_true), predictions)


def test_cross_entropy_batch():
    num_classes = 4
    batch_size = 3

    predictions = np.random.randint(-1, 3, size=(batch_size, num_classes)).astype(np.float64)
    y_true = np.random.randint(0, num_classes, size=(batch_size, 1)).astype(np.int32)

    criterion = CrossEntropyLoss()

    # It's basically correct but error occurred due to division gradient by batch_size in loss function
    GradientCheck.check_gradient(lambda y_pred: criterion(y_pred, y_true), predictions)
