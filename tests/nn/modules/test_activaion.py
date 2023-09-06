import numpy as np

import torchy.nn as nn
from tests.gradient_check import eval_numerical_gradient_array
from tests.utils import rel_error


def test_relu_forward():
    # Test the relu_forward function

    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

    relu = nn.ReLU()
    out = relu(x)
    correct_out = np.array([[0., 0., 0., 0., ],
                            [0., 0., 0.04545455, 0.13636364, ],
                            [0.22727273, 0.31818182, 0.40909091, 0.5, ]])

    # Compare your output with ours. The error should be on the order of e-8
    assert rel_error(out, correct_out) <= 1e-7


def test_relu_backward():
    np.random.seed(231)
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    relu = nn.ReLU()
    dx_num = eval_numerical_gradient_array(lambda x: relu(x), x, dout)

    relu(x)
    dx = relu.backward(dout)

    assert rel_error(dx_num, dx) <= 1e-11
