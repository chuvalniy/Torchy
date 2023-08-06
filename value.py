import numpy as np


class Value:
    def __init__(self, value):
        self.data = value
        self.grad = np.zeros_like(value)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
