import numpy as np

from gradient_check import GradientCheck
from layer import ReLU

if __name__ == "__main__":
    X = np.array([[1, -2, 3],
                  [-1, 2, 0.1]
                  ])

    assert GradientCheck.check_layer_gradient(ReLU(), X)
