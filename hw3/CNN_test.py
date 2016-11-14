#!/usr/bin/env python3

from sys import argv, stderr
import pickle
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.backend import set_image_dim_ordering
from keras.optimizers import SGD, Adagrad

def die(msg):
    print(msg, file = stderr)
    exit(1)

def unFlatten(V):
    Result = []
    for i in range(0, 1024, 32):
        Result.append(V[i:i+32])
    return np.array(Result)

def processTest(Data):
    ID = []
    Result = []
    for i in range(10000):
        ID.append(Data['ID'][i])
        Datum = Data['data'][i]
        Result.append([unFlatten(e) for e in
                        [Datum[:1024], Datum[1024:2048], Datum[2048:3072]]
                        ])
    return ID, np.array(Result)

def main():
    if len(argv) != 4:
        die('Usage: {} [test data] [model] [output]'
            .format(argv[0]))

    set_image_dim_ordering('th')

    TestID, TestData = processTest(pickle.load(open(argv[1], 'rb')))
    TestData = TestData / 255.0

    Model = keras.models.load_model(argv[2])
    TestLabel = Model.predict(TestData)

    Output = open(argv[3], 'w')
    Output.write("ID,class\n")
    for i in range(10000):
        Label = 0
        for j in range(1, 10):
            if TestLabel[i][j] > TestLabel[i][Label]:
                Label = j
        Output.write("{},{}\n".format(TestID[i], Label))
    Output.close()
    return 0

if __name__ == '__main__': main()
