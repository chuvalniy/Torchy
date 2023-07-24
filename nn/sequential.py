from layer import Module
import copy


class Sequential:
    def __init__(self, *args):
        self._layers = []

        for arg in args:
            if not isinstance(arg, Module):
                raise Exception("Invalid argument for layer")

            self._layers.append(arg)

    def __call__(self, X):
        copy_X = copy.deepcopy(X)
        for layer in self._layers:
            copy_X = layer(copy_X)

        return copy_X

    def predict(self, X):
        pass
