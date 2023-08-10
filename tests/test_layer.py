import numpy as np

from layer import Linear, ReLU, BatchNorm1d
from tests.gradient_check import GradientCheck
from sklearn.datasets import make_classification

def test_linear():
    X = np.array([[1, -2, 3],
                  [-1, 2, 0.1]
                  ])

    assert GradientCheck.check_layer_gradient(Linear(3, 4), X)
    assert GradientCheck.check_layer_param_gradient(Linear(3, 4), X, 'W')
    assert GradientCheck.check_layer_param_gradient(Linear(3, 4), X, 'B')


def test_relu():
    X = np.array([[1, -2, 3],
                  [-1, 2, 0.1]
                  ])

    assert GradientCheck.check_layer_gradient(ReLU(), X)


def test_batchnorm1d():
    X, _ = make_classification(n_samples=50, n_features=5, n_redundant=0)

    assert GradientCheck.check_layer_param_gradient(BatchNorm1d(5), X, 'beta')
    assert GradientCheck.check_layer_param_gradient(BatchNorm1d(5), X, 'gamma')
    assert GradientCheck.check_layer_gradient(BatchNorm1d(5), X)
