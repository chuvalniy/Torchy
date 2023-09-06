import numpy as np

import torchy.nn as nn
from tests.gradient_check import eval_numerical_gradient_array
from tests.utils import rel_error


def test_maxpool2d_forward():
    x_shape = (2, 3, 4, 4)
    x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)

    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    out = pool(x)

    correct_out = np.array([[[[-0.26315789, -0.24842105],
                              [-0.20421053, -0.18947368]],
                             [[-0.14526316, -0.13052632],
                              [-0.08631579, -0.07157895]],
                             [[-0.02736842, -0.01263158],
                              [0.03157895, 0.04631579]]],
                            [[[0.09052632, 0.10526316],
                              [0.14947368, 0.16421053]],
                             [[0.20842105, 0.22315789],
                              [0.26736842, 0.28210526]],
                             [[0.32631579, 0.34105263],
                              [0.38526316, 0.4]]]])

    assert rel_error(out, correct_out) <= 1e-7


def test_maxpool2d_backward():
    np.random.seed(231)
    x = np.random.randn(3, 2, 8, 8)
    dout = np.random.randn(3, 2, 4, 4)

    pool = nn.MaxPool2d(kernel_size=2, stride=2)

    dx_num = eval_numerical_gradient_array(lambda x: pool(x), x, dout)
    dx = pool.backward(dout)

    assert rel_error(dx, dx_num) <= 1e-11
