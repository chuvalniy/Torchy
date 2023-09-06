import numpy as np

import torchy.nn as nn
from tests.gradient_check import eval_numerical_gradient_array
from tests.utils import rel_error


def test_conv2d_backward():
    np.random.seed(231)
    x = np.random.randn(4, 3, 5, 5)
    w = np.random.randn(2, 3, 3, 3)
    b = np.random.randn(2, )
    dout = np.random.randn(4, 2, 5, 5)

    conv = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)
    conv.weight.data = w
    conv.bias.data = b

    dx_num = eval_numerical_gradient_array(lambda x: conv(x), x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: conv(x), w, dout)
    db_num = eval_numerical_gradient_array(lambda b: conv(x), b, dout)

    conv(x)
    dx = conv.backward(dout)

    assert rel_error(dx, dx_num) <= 1e-8
    assert rel_error(conv.weight.grad, dw_num) <= 1e-8
    assert rel_error(conv.bias.grad, db_num) <= 1e-8


def test_conv2d_forward():
    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=3)

    layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1)
    layer.weight.data = w
    layer.bias.data = b
    out = layer(x)
    correct_out = np.array([[[[-0.08759809, -0.10987781],
                              [-0.18387192, -0.2109216]],
                             [[0.21027089, 0.21661097],
                              [0.22847626, 0.23004637]],
                             [[0.50813986, 0.54309974],
                              [0.64082444, 0.67101435]]],
                            [[[-0.98053589, -1.03143541],
                              [-1.19128892, -1.24695841]],
                             [[0.69108355, 0.66880383],
                              [0.59480972, 0.56776003]],
                             [[2.36270298, 2.36904306],
                              [2.38090835, 2.38247847]]]])

    assert rel_error(out, correct_out) <= 1e-7
