import numpy as np

from layer import Linear
from optim import SGD
from sequential import Sequential
from value import Value


def get_parameters():
    param_1 = Value(np.array([[1, 2, 3], [4, 5, 6]]))
    param_1.grad = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    d = {"param_1": param_1}

    return d


def test_zero_grad_custom_params():
    params = get_parameters()

    optimizer = SGD(params, lr=1e-4, weight_decay=0.0)

    optimizer.zero_grad()

    for param_name, param in optimizer._params.items():
        assert np.sum(param.grad) == 0.0
