import numpy as np

import torchy.nn as nn
from tests.gradient_check import eval_numerical_gradient_array
from tests.utils import rel_error


def test_vanilla_rnn_forward():
    N, T, D, H = 2, 3, 4, 5

    x = np.linspace(-0.1, 0.3, num=N * T * D).reshape(N, T, D)
    h0 = np.linspace(-0.3, 0.1, num=N * H).reshape(N, H)
    Wx = np.linspace(-0.2, 0.4, num=D * H).reshape(D, H)
    Wh = np.linspace(-0.4, 0.1, num=H * H).reshape(H, H)
    b = np.linspace(-0.7, 0.1, num=H)

    rnn = nn.RNN(N, H)
    rnn.weight_xh.data = Wx
    rnn.weight_hh.data = Wh
    rnn.bias.data = b
    output, _ = rnn(x, h0)
    expected_h = np.asarray([
        [
            [-0.42070749, -0.27279261, -0.11074945, 0.05740409, 0.22236251],
            [-0.39525808, -0.22554661, -0.0409454, 0.14649412, 0.32397316],
            [-0.42305111, -0.24223728, -0.04287027, 0.15997045, 0.35014525],
        ],
        [
            [-0.55857474, -0.39065825, -0.19198182, 0.02378408, 0.23735671],
            [-0.27150199, -0.07088804, 0.13562939, 0.33099728, 0.50158768],
            [-0.51014825, -0.30524429, -0.06755202, 0.17806392, 0.40333043]]])

    assert rel_error(expected_h, output) <= 1e-7


def test_rnn_vanilla_backward():
    np.random.seed(231)

    N, D, T, H = 2, 3, 10, 5

    x = np.random.randn(N, T, D)
    h0 = np.random.randn(N, H)
    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(H)

    rnn = nn.RNN(D, H)
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
