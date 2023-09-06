import numpy as np

import torchy.nn as nn
from tests.gradient_check import eval_numerical_gradient_array
from tests.utils import rel_error, print_mean_std

def test_batchnorm1d_train_forward():
    np.random.seed(231)
    N, D1, D2, D3 = 200, 50, 60, 3
    X = np.random.randn(N, D1)
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    a = np.maximum(0, X.dot(W1)).dot(W2)

    print('Before batch normalization:')
    print_mean_std(a, axis=0)

    gamma = np.ones((D3,))
    beta = np.zeros((D3,))

    print('After batch normalization (gamma=1, beta=0)')
    batchnorm = nn.BatchNorm1d(D3)
    batchnorm.gamma.data = gamma
    batchnorm.beta.data = beta
    a_norm = batchnorm(a)
    print_mean_std(a_norm, axis=0)

    assert rel_error(a_norm.mean(axis=0), beta) <= 1e-8
    assert rel_error(a_norm.std(axis=0), gamma) <= 1e-8

    gamma = np.asarray([1.0, 2.0, 3.0])
    beta = np.asarray([11.0, 12.0, 13.0])

    batchnorm = nn.BatchNorm1d(D3)
    batchnorm.gamma.data = gamma
    batchnorm.beta.data = beta
    print('After batch normalization (gamma=', gamma, ', beta=', beta, ')')
    a_norm = batchnorm(a)
    print_mean_std(a_norm, axis=0)

    assert rel_error(a_norm.mean(axis=0), beta) <= 1e-8
    assert rel_error(a_norm.std(axis=0), gamma) <= 1e-8

def test_batchnorm1d_test_forward():
    # Check the test-time forward pass by running the training-time
    # forward pass many times to warm up the running averages, and then
    # checking the means and variances of activations after a test-time
    # forward pass.

    np.random.seed(231)
    N, D1, D2, D3 = 200, 50, 60, 3
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)

    gamma = np.ones(D3)
    beta = np.zeros(D3)

    batchnorm = nn.BatchNorm1d(D3)
    batchnorm.gamma.data = gamma
    batchnorm.beta.data = beta
    for t in range(50):
        X = np.random.randn(N, D1)
        a = np.maximum(0, X.dot(W1)).dot(W2)
        batchnorm(a)

    batchnorm._train = False
    X = np.random.randn(N, D1)
    a = np.maximum(0, X.dot(W1)).dot(W2)
    a_norm = batchnorm(a)

    # Means should be close to zero and stds close to one, but will be
    # noisier than training-time forward passes.
    print('After batch normalization (test-time):')
    print_mean_std(a_norm, axis=0)


def test_batchnorm2d_train_forward():
    np.random.seed(231)

    # Check the training-time forward pass by checking means and variances
    # of features both before and after spatial batch normalization.
    N, C, H, W = 2, 3, 4, 5
    x = 4 * np.random.randn(N, C, H, W) + 10

    print('Before spatial batch normalization:')
    print('  shape: ', x.shape)
    print('  means: ', x.mean(axis=(0, 2, 3)))
    print('  stds: ', x.std(axis=(0, 2, 3)))

    # Means should be close to zero and stds close to one
    gamma, beta = np.ones(C), np.zeros(C)
    batch_norm = nn.BatchNorm2d(n_output=C)
    batch_norm.gamma.data = gamma
    batch_norm.beta.data = beta
    out = batch_norm(x)
    print('After spatial batch normalization:')
    print('  shape: ', out.shape)
    print('  means: ', out.mean(axis=(0, 2, 3)))
    print('  stds: ', out.std(axis=(0, 2, 3)))

    assert rel_error(out.mean(axis=(0, 2, 3)), beta) <= 1e-6
    assert rel_error(out.std(axis=(0, 2, 3)), gamma) <= 1e-6

    # Means should be close to beta and stds close to gamma
    gamma, beta = np.asarray([3, 4, 5]), np.asarray([6, 7, 8])
    batch_norm = nn.BatchNorm2d(n_output=C)
    batch_norm.gamma.data = gamma
    batch_norm.beta.data = beta
    out = batch_norm(x)
    print('After spatial batch normalization (nontrivial gamma, beta):')
    print('  shape: ', out.shape)
    print('  means: ', out.mean(axis=(0, 2, 3)))
    print('  stds: ', out.std(axis=(0, 2, 3)))

    assert rel_error(out.mean(axis=(0, 2, 3)), beta) <= 1e-6
    assert rel_error(out.std(axis=(0, 2, 3)), gamma) <= 1e-6


def test_batchnorm2d_test_forward():
    np.random.seed(231)

    # Check the test-time forward pass by running the training-time
    # forward pass many times to warm up the running averages, and then
    # checking the means and variances of activations after a test-time
    # forward pass.
    N, C, H, W = 10, 4, 11, 12

    gamma = np.ones(C)
    beta = np.zeros(C)
    batch_norm = nn.BatchNorm2d(n_output=C)
    batch_norm.gamma.data = gamma
    batch_norm.beta.data = beta

    for t in range(50):
        x = 2.3 * np.random.randn(N, C, H, W) + 13
        batch_norm(x)

    x = 2.3 * np.random.randn(N, C, H, W) + 13
    batch_norm._train = False
    out = batch_norm(x)

    # Means should be close to zero and stds close to one, but will be
    # noisier than training-time forward passes.
    print('After spatial batch normalization (test-time):')
    print('  means: ', out.mean(axis=(0, 2, 3)))
    print('  stds: ', out.std(axis=(0, 2, 3)))

def test_batchnorm1d_backward():
    np.random.seed(231)
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    batchnorm = nn.BatchNorm1d(D)
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

    batchnorm = nn.BatchNorm2d(C)
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
