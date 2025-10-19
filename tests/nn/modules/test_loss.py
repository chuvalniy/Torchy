import numpy as np
import torchy.nn as nn
from tests.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from tests.utils import rel_error


def test_softmax():
    np.random.seed(231)
    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)

    dx_num = eval_numerical_gradient(lambda x: nn.CrossEntropyLoss()(x, y)[0], x, verbose=False)
    loss, dx = nn.CrossEntropyLoss()(x, y)

    # Test softmax_loss function. Loss should be close to 2.3 and dx error should be around e-8
    print('\nTesting softmax_loss:')
    print('loss: ', loss)
    assert rel_error(dx_num, dx) < 1e-8


def test_kldiv():
    np.random.seed(231)
    x = 1e-3 * np.random.randn(10, 10)
    y = 1e-3 * np.random.randn(10, 10)

    x = nn.compute_softmax(x)
    y = nn.compute_softmax(y)

    dx_num = eval_numerical_gradient(lambda x: nn.KLDivLoss()(x, y)[0], x, verbose=False)
    loss, dx = nn.KLDivLoss()(x, y)

    print('\nTesting kl_div_loss:')
    print('loss: ', loss)
    assert rel_error(dx_num, dx) < 1e-8