import numpy as np

from tests.gradient_check import eval_numerical_gradient_array
from tests.utils import rel_error
from torchy.module import Conv2d, MaxPool2d, BatchNorm2d, Dropout, BatchNorm1d, Linear, ReLU, RNN


def test_linear_backward():
    np.random.seed(231)
    x = np.random.randn(10, 2, 3)
    w = np.random.randn(6, 5)
    b = np.random.randn(5)
    dout = np.random.randn(10, 5)

    linear = Linear(x.shape[0], w.shape[1])
    linear.weight.data = w
    linear.bias.data = b

    x = np.reshape(x, (x.shape[0], -1))

    dx_num = eval_numerical_gradient_array(lambda x: linear(x), x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: linear(x), w, dout)
    db_num = eval_numerical_gradient_array(lambda b: linear(x), b, dout)

    linear(x)
    dx = linear.backward(dout)

    assert rel_error(dx_num, dx) <= 1e-10
    assert rel_error(dw_num, linear.weight.grad) <= 1e-10
    assert rel_error(db_num, linear.bias.grad) <= 1e-10


def test_relu_backward():
    np.random.seed(231)
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    relu = ReLU()
    dx_num = eval_numerical_gradient_array(lambda x: relu(x), x, dout)

    relu(x)
    dx = relu.backward(dout)

    assert rel_error(dx_num, dx) <= 1e-11


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

    conv(x)
    dx = conv.backward(dout)

    assert rel_error(dx, dx_num) <= 1e-8
    assert rel_error(conv.weight.grad, dw_num) <= 1e-8
    assert rel_error(conv.bias.grad, db_num) <= 1e-8


def test_rnn_vanilla_backward():
    np.random.seed(231)

    N, D, T, H = 2, 3, 10, 5

    x = np.random.randn(N, T, D)
    h0 = np.random.randn(N, H)
    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(H)

    rnn = RNN(D, H)
    rnn.weight_xh.data = Wx
    rnn.weight_hh.data = Wh
    rnn.bias.data = b
    out, _ = rnn(x, h0)

    dout = np.random.randn(*out.shape)

    dx = rnn.backward(dout)

    fx = lambda x: rnn(x, h0)[0]
    fWx = lambda Wx: rnn(x, h0)[0]
    fWh = lambda Wh: rnn(x, h0)[0]
    fb = lambda b: rnn(x, h0)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
    dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
    db_num = eval_numerical_gradient_array(fb, b, dout)

    assert rel_error(dx_num, dx) <= 1e-7
    assert rel_error(dWx_num, rnn.weight_xh.grad) <= 1e-7
    assert rel_error(dWh_num, rnn.weight_hh.grad) <= 1e-7
    assert rel_error(db_num, rnn.bias.grad) <= 1e-7


def test_maxpool2d_backward():
    np.random.seed(231)
    x = np.random.randn(3, 2, 8, 8)
    dout = np.random.randn(3, 2, 4, 4)

    pool = MaxPool2d(kernel_size=2, stride=2)

    dx_num = eval_numerical_gradient_array(lambda x: pool(x), x, dout)
    dx = pool.backward(dout)

    assert rel_error(dx, dx_num) <= 1e-11


def test_batchnorm1d_backward():
    np.random.seed(231)
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    batchnorm = BatchNorm1d(D)
    batchnorm.gamma.data = gamma
    batchnorm.beta.data = beta

    fx = lambda x: batchnorm(x)
    fg = lambda a: batchnorm(x)
    fb = lambda b: batchnorm(x)

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    da_num = eval_numerical_gradient_array(fg, gamma, dout)
    db_num = eval_numerical_gradient_array(fb, beta, dout)

    batchnorm(x)
    dx = batchnorm.backward(dout)

    assert rel_error(dx_num, dx) < 1e-8
    assert rel_error(da_num, batchnorm.gamma.grad) < 1e-11
    assert rel_error(db_num, batchnorm.beta.grad) < 1e-11


def test_batchnorm2d_backward():
    np.random.seed(231)
    N, C, H, W = 2, 3, 4, 5
    x = 5 * np.random.randn(N, C, H, W) + 12
    gamma = np.random.randn(C)
    beta = np.random.randn(C)
    dout = np.random.randn(N, C, H, W)

    batchnorm = BatchNorm2d(C)
    batchnorm.gamma.data = gamma
    batchnorm.beta.data = beta

    fx = lambda x: batchnorm(x)
    fg = lambda a: batchnorm(x)
    fb = lambda b: batchnorm(x)

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    da_num = eval_numerical_gradient_array(fg, gamma, dout)
    db_num = eval_numerical_gradient_array(fb, beta, dout)

    batchnorm(x)
    dx = batchnorm.backward(dout)

    assert rel_error(dx_num, dx) <= 1e-5
    assert rel_error(da_num, batchnorm.gamma.grad) <= 1e-11
    assert rel_error(db_num, batchnorm.beta.grad) <= 1e-11


def test_dropout_backward():
    np.random.seed(231)
    x = np.random.randn(10, 10) + 10
    dout = np.random.randn(*x.shape)

    dropout = Dropout(0.2, seed=123)
    dropout(x)
    dx = dropout.backward(dout)
    dx_num = eval_numerical_gradient_array(lambda xx: Dropout(0.2, seed=123)(xx), x, dout)

    assert rel_error(dx, dx_num) <= 1e-10
