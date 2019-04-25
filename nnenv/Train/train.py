import numpy as np
from Initialize_Weights import initialize_weights as ini
from Predict import  predict as pred
from Sigmoid_Function import sigmoid as sig


def train(self, file, X, y, iterations):
    dim = file.shape
    ini.BasicNuralNetwork.rand_weight = 2 * np.random.random(dim[1] - 1 ,1) -1

    for i in range(iterations):
        output = pred.predict(X)
        error = y - output

        adjustment = np.dot(X.T, error * sig.__sigmoid(output,derive=True) )
        ini.BasicNuralNetwork.rand_weight += adjustment