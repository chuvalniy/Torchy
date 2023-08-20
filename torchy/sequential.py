import copy

import numpy as np

from layer import Layer, NnLayer
from value import Value


class Sequential:
    """
    Module for structuring neural network layers
    """

    def __init__(self, *args: Layer):
        """
        :param args: Layer - collection of neural network layers that will be executed sequentially
        """
        self._layers: list[Layer] = []

        for arg in args:
            if not isinstance(arg, Layer):
                print(type(arg))
                raise Exception("Invalid argument for layer")

            self._layers.append(arg)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Iterates over a list of neural network layers and
        sequentially performs a forward pass

        :param x: numpy array (n-dimensional) - data for prediction
        :return: numpy array (n-dimensional) - logits after performing forward pass
        """
        copy_X = copy.deepcopy(x)
        for layer in self._layers:
            copy_X = layer(copy_X)

        return copy_X

    def backward(self, d_out: np.ndarray):
        """
        Iterates over a reversed list of neural network layers
        and calculates backward pass for each of these layers

        :param d_out: numpy array (n-dimensional) - gradient of loss function
        """
        d_out_copy = copy.deepcopy(d_out)
        for layer in reversed(self._layers):
            d_out_copy = layer.backward(d_out_copy)

    def predict(self, x):
        pass

    def params(self) -> dict[str, Value]:
        """
        Passes through every layer of the neural network and gathers their
        parameters into dictionary

        :return: dict (param_name: str, param: Param)
        """
        d = {}

        for idx, layer in enumerate(self._layers):
            if isinstance(layer, NnLayer):
                for param_name, param in layer.params().items():
                    d[f"{param_name}_{idx}"] = param

        return d
