import numpy as np

import torchy.nn as nn
from tests.gradient_check import eval_numerical_gradient_array
from tests.utils import rel_error


def test_dropout_backward():
    np.random.seed(231)
    x = np.random.randn(10, 10) + 10
    dout = np.random.randn(*x.shape)

    dropout = nn.Dropout(0.2, seed=123)
    dropout(x)
    dx = dropout.backward(dout)
    dx_num = eval_numerical_gradient_array(lambda xx: nn.Dropout(0.2, seed=123)(xx), x, dout)

    assert rel_error(dx, dx_num) <= 1e-10

def test_dropout_forward():
    np.random.seed(231)
    x = np.random.randn(500, 500) + 10

    for p in [0.25, 0.4, 0.7]:
        dropout = nn.Dropout(p)
        out = dropout(x)

        dropout._train = False
        out_test = dropout(x)

        print('Running tests with p = ', p)
        print('Mean of input: ', x.mean())
        print('Mean of train-time output: ', out.mean())
        print('Mean of test-time output: ', out_test.mean())
        print('Fraction of train-time output set to zero: ', (out == 0).mean())
        print('Fraction of test-time output set to zero: ', (out_test == 0).mean())
        print()