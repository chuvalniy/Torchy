import numpy as np
from value import Value
import copy
from abc import ABC

class Module(ABC):
    def zero_grad(self):
        pass

    def params(self):
        pass



class Linear(Module):
    def __init__(self, in_channels, out_channels):
        self.W = Value(in_channels, out_channels)
        self.B = Value(in_channels, 1)

        self.X = None

    def forward(self, X):
        self.X = copy.deepcopy(X)

        return self.X * self.W.data + self.B.data

    def backward(self, d_out):
        pass
