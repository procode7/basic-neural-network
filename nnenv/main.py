import pandas as pd
import numpy as np

from Initialize_Weights import initialize_weights as init
from Train import  train as train
from Predict import predict as predict


if __name__ == "__main__":
    data =pd.read_csv("file.csv")
    print(len(data))

    X = data.iloc[:,0:4].values
    y = data.iloc[:,[4]].values
    number_of_iterations = 6000
    clf = init.BasicNuralNetwork()

    #Training
    train.train(data,X,y,number_of_iterations)

    #predict
    prediction = np.array([0, 1, 1, 0])
    res = predict.predict(prediction)[0]

    #Final Output
    if res >= 0.5:
        print("Prediction:",1)
    else:
        print("Prediction:",0)