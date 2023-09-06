import numpy as np
import torchy.nn as nn
from tests.gradient_check import eval_numerical_gradient
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
