import numpy as np


def kaiming_init(n_input: int) -> float:
    """
    Computes Kaiming Initialization

    :param n_input: int - size of input parameters of neural network layer
    :return: float - result of Kaiming Initialization
    """

    return 1 / np.sqrt(n_input / 2)
