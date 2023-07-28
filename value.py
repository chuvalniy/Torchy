import numpy as np
import random


class Value:
    def __init__(self, value):
        self.data = value
        self.grad = np.zeros_like(value)