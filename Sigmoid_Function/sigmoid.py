import numpy as np


def __sigmoid(self, x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
