import numpy as np

from tests.gradient_check import eval_numerical_gradient_array
from tests.utils import rel_error
from torchy.layer import Conv2d, MaxPool2d


def test_conv2d_backward():
    np.random.seed(231)
    x = np.random.randn(4, 3, 5, 5)
    w = np.random.randn(2, 3, 3, 3)
    b = np.random.randn(2, )
    dout = np.random.randn(4, 2, 5, 5)

    conv = Conv2d(3, 2, kernel_size=3, stride=1, padding=1)
    conv.weight.data = w
    conv.bias.data = b

    dx_num = eval_numerical_gradient_array(lambda x: conv(x), x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: conv(x), w, dout)
    db_num = eval_numerical_gradient_array(lambda b: conv(x), b, dout)

    _ = conv(x)
    dx = conv.backward(dout)

    assert rel_error(dx, dx_num) <= 1e-8
    assert rel_error(conv.weight.grad, dw_num) <= 1e-8
    assert rel_error(conv.bias.grad, db_num) <= 1e-8


def test_maxpool2d_backward():
    np.random.seed(231)
    x = np.random.randn(3, 2, 8, 8)
    dout = np.random.randn(3, 2, 4, 4)

    pool = MaxPool2d(kernel_size=2, stride=2)

    dx_num = eval_numerical_gradient_array(lambda x: pool(x), x, dout)
    dx = pool.backward(dout)

    assert  rel_error(dx, dx_num) <= 1e-11
