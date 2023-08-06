import numpy as np

from layer import ReLU
from tests.gradient_check import GradientCheck


def test_relu():
    X = np.array([[1, -2, 3],
                  [-1, 2, 0.1]
                  ])

    assert GradientCheck.check_layer_gradient(ReLU(), X)
