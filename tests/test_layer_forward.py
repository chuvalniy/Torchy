import numpy as np

from tests.utils import rel_error, print_mean_std
from torchy.module import Conv2d, MaxPool2d, BatchNorm2d, Dropout, BatchNorm1d, Linear, ReLU, RNN, Embedding


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
    linear = Linear(input_size, output_dim)
    linear.weight.data = w
    linear.bias.data = b

    out = linear(x)
    correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                            [3.25553199, 3.5141327, 3.77273342]])

    assert rel_error(out, correct_out) <= 1e-9


def test_relu_forward():
    # Test the relu_forward function

    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

    relu = ReLU()
    out = relu(x)
    correct_out = np.array([[0., 0., 0., 0., ],
                            [0., 0., 0.04545455, 0.13636364, ],
                            [0.22727273, 0.31818182, 0.40909091, 0.5, ]])

    # Compare your output with ours. The error should be on the order of e-8
    assert rel_error(out, correct_out) <= 1e-7


def test_conv2d_forward():
    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=3)

    layer = Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1)
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


def test_vanilla_rnn_forward():
    N, T, D, H = 2, 3, 4, 5

    x = np.linspace(-0.1, 0.3, num=N * T * D).reshape(N, T, D)
    h0 = np.linspace(-0.3, 0.1, num=N * H).reshape(N, H)
    Wx = np.linspace(-0.2, 0.4, num=D * H).reshape(D, H)
    Wh = np.linspace(-0.4, 0.1, num=H * H).reshape(H, H)
    b = np.linspace(-0.7, 0.1, num=H)

    rnn = RNN(N, H)
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


def test_embedding_forward():
    N, T, V, D = 2, 4, 5, 3

    x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
    W = np.linspace(0, 1, num=V * D).reshape(V, D)

    emb = Embedding(V, D)
    emb.weight.data = W

    out = emb(x)
    expected_out = np.asarray([
        [[0., 0.07142857, 0.14285714],
         [0.64285714, 0.71428571, 0.78571429],
         [0.21428571, 0.28571429, 0.35714286],
         [0.42857143, 0.5, 0.57142857]],
        [[0.42857143, 0.5, 0.57142857],
         [0.21428571, 0.28571429, 0.35714286],
         [0., 0.07142857, 0.14285714],
         [0.64285714, 0.71428571, 0.78571429]]])

    assert rel_error(expected_out, out) <= 1e-7


def test_maxpool2d_forward():
    x_shape = (2, 3, 4, 4)
    x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)

    pool = MaxPool2d(kernel_size=2, stride=2)
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
    batchnorm = BatchNorm1d(D3)
    batchnorm.gamma.data = gamma
    batchnorm.beta.data = beta
    a_norm = batchnorm(a)
    print_mean_std(a_norm, axis=0)

    assert rel_error(a_norm.mean(axis=0), beta) <= 1e-8
    assert rel_error(a_norm.std(axis=0), gamma) <= 1e-8

    gamma = np.asarray([1.0, 2.0, 3.0])
    beta = np.asarray([11.0, 12.0, 13.0])

    batchnorm = BatchNorm1d(D3)
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

    batchnorm = BatchNorm1d(D3)
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
    batch_norm = BatchNorm2d(n_output=C)
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
    batch_norm = BatchNorm2d(n_output=C)
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
    batch_norm = BatchNorm2d(n_output=C)
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


def test_dropout_forward():
    np.random.seed(231)
    x = np.random.randn(500, 500) + 10

    for p in [0.25, 0.4, 0.7]:
        dropout = Dropout(p)
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
