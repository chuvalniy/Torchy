import numpy as np
import random


class Value:
    def __init__(self, in_num, out_num):
        self.data = np.ones(shape=(in_num, out_num)) * (1 / np.sqrt(in_num) * random.uniform(in_num, out_num))  # Xavier
        self.grad = np.zeros(shape=(in_num, out_num))
