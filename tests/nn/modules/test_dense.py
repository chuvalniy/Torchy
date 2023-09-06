import numpy as np

import torchy.nn as nn
from tests.gradient_check import eval_numerical_gradient_array
from tests.utils import rel_error


def test_linear_forward():
    # Test the affine_forward function

    num_inputs = 2
    input_shape = (4, 5, 6)
    output_dim = 3

    input_size = num_inputs * np.prod(input_shape)
    weight_size = output_dim * np.prod(input_shape)

    x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
    w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
    b = np.linspace(-0.3, 0.1, num=output_dim)

    x = np.reshape(x, (x.shape[0], -1))
    linear = nn.Linear(input_size, output_dim)
    linear.weight.data = w
    linear.bias.data = b

    out = linear(x)
    correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                            [3.25553199, 3.5141327, 3.77273342]])

    assert rel_error(out, correct_out) <= 1e-9


def test_linear_backward():
    np.random.seed(231)
    x = np.random.randn(10, 2, 3)
    w = np.random.randn(6, 5)
    b = np.random.randn(5)
    dout = np.random.randn(10, 5)

    linear = nn.Linear(x.shape[0], w.shape[1])
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
