import numpy as np

import torchy.nn as nn
from tests.gradient_check import eval_numerical_gradient_array
from tests.utils import rel_error


def test_embedding_forward():
    N, T, V, D = 2, 4, 5, 3

    x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
    W = np.linspace(0, 1, num=V * D).reshape(V, D)

    emb = nn.Embedding(V, D)
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