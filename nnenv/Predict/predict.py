import numpy as np
from Initialize_Weights import initialize_weights as ini


def predict(self,x):
    predicted = np.dot(x,ini.BasicNuralNetwork.rand_weight)