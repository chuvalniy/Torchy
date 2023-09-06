import numpy as np

from torchy.value import Value


def kaiming_uniform(shape: tuple[int, ...], nonlinearity: str = "relu") -> Value:
    gain = _calculate_gain(nonlinearity)
    fan_in = _calculate_fan_in(shape)
    std = gain / np.sqrt(fan_in)
    bound = std * np.sqrt(3.0)
    return Value(np.random.uniform(-bound, bound, size=shape))


def normal(shape: tuple[int, ...]) -> Value:
    return Value(np.random.randn(*shape))


def kaiming_normal(shape: tuple[int, ...], nonlinearity: str = "relu") -> Value:
    gain = _calculate_gain(nonlinearity)
    fan_in = _calculate_fan_in(shape)
    std = gain / np.sqrt(fan_in)
    return Value(std * np.random.randn(*shape))


def _calculate_gain(nonlinearity: str) -> float:
    if nonlinearity == 'relu':
        return np.sqrt(2.0)
    elif nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    else:
        raise NotImplementedError(f"Gain calculation for '{nonlinearity}' non-linearity is not implemented.")


def _calculate_fan_in(shape: tuple[int, ...]) -> int:
    if len(shape) == 2 or len(shape) == 1:
        return shape[0]
    elif len(shape) == 4:
        return shape[1] * shape[2] * shape[3]
    else:
        raise NotImplementedError(f"Fan in calculation for {shape} shape is not implemented.")
