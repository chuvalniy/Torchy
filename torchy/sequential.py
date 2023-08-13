from layer import Layer, NnLayer
import copy

from value import Value


class Sequential:
    def __init__(self, *args: Layer):
        self._layers = []

        for arg in args:
            if not isinstance(arg, Layer):
                raise Exception("Invalid argument for layer")

            self._layers.append(arg)

    def __call__(self, X):
        copy_X = copy.deepcopy(X)
        for layer in self._layers:
            copy_X = layer(copy_X)

        return copy_X

    def backward(self, d_out):
        d_out_copy = copy.deepcopy(d_out)
        for layer in reversed(self._layers):
            d_out_copy = layer.backward(d_out_copy)

    def predict(self, X):
        pass

    def params(self) -> dict[str, Value]:
        d = {}

        for idx, layer in enumerate(self._layers):
            if isinstance(layer, NnLayer):
                for param_name, param in layer.params().items():
                    d[f"{param_name}_{idx}"] = param

        return d
