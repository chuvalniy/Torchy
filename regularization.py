import numpy as np


class L2:
    def __init__(self, weight_decay=0.0):
        self.weight_decay = weight_decay

    def l2_loss(self, params):
        l2_loss = 0.0
        for param in params.values():
            l2_loss += self.weight_decay * np.sum(np.square(param.data))

        return l2_loss

