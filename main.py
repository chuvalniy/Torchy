import numpy as np

from gradient_check import GradientCheck
from layer import ReLU, Linear


def test_relu():
    X = np.array([[1, -2, 3],
                  [-1, 2, 0.1]
                  ])

    assert GradientCheck.check_layer_gradient(ReLU(), X)


def test_dense():
    X = np.array([[1, -2, 3],
                  [-1, 2, 0.1]
                  ])

    assert GradientCheck.check_layer_gradient(Linear(3, 4), X)
    assert GradientCheck.check_layer_param_gradient(Linear(3, 4), X, 'W')
    assert GradientCheck.check_layer_param_gradient(Linear(3, 4), X, 'B')



if __name__ == "__main__":
    test_dense()




