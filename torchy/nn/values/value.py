import numpy as np


class Value:
    """
    Trainable parameter
    Contain both data and gradient values
    """

    def __init__(self, value: np.ndarray):
        """
        :param value: numpy array (n-dimensional) - layer parameters
        """
        self.data = value
        self.grad = np.zeros_like(value)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
