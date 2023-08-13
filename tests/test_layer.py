import numpy as np
from sklearn.datasets import make_classification

from layer import Linear, ReLU, BatchNorm1d, Conv2d, MaxPool2d
from tests.gradient_check import GradientCheck


def test_linear():
    X = np.array([[1, -2, 3],
                  [-1, 2, 0.1]
                  ])

    assert GradientCheck.check_layer_gradient(Linear(3, 4), X)
    assert GradientCheck.check_layer_param_gradient(Linear(3, 4), X, 'W')
    assert GradientCheck.check_layer_param_gradient(Linear(3, 4), X, 'B')


def test_conv2d():
    X = np.random.randn(2, 2, 7, 7)

    layer = Conv2d(in_channels=2, out_channels=2, kernel_size=2, padding=0)
    result = layer.forward(X)
    d_input = layer.backward(np.ones_like(result))
    assert d_input.shape == X.shape

    layer = Conv2d(in_channels=2, out_channels=2, kernel_size=2, padding=0)
    assert GradientCheck.check_layer_gradient(layer, X)

    layer = Conv2d(in_channels=2, out_channels=2, kernel_size=2, padding=0)
    assert GradientCheck.check_layer_param_gradient(layer, X, 'W')

    layer = Conv2d(in_channels=2, out_channels=2, kernel_size=2, padding=0)
    assert GradientCheck.check_layer_param_gradient(layer, X, 'B')


def test_conv2_with_padding():
    X = np.random.randn(2, 2, 7, 7)

    layer = Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
    result = layer.forward(X)
    assert result.shape == X.shape, "Result shape: %s - Expected shape %s" % (result.shape, X.shape)
    d_input = layer.backward(np.ones_like(result))
    assert d_input.shape == X.shape
    layer = Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
    assert GradientCheck.check_layer_gradient(layer, X)


def test_maxpool2d():
    X = np.random.randn(2, 2, 7, 7)

    pool = MaxPool2d(kernel_size=2, stride=2)
    result = pool.forward(X)
    assert result.shape == (2, 2, 3, 3)

    assert GradientCheck.check_layer_gradient(pool, X)


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
