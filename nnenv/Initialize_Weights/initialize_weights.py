import numpy as np


class BasicNuralNetwork(object):
    def __init__(self):
        np.random.seed(1234)
        self.rand_weight = []