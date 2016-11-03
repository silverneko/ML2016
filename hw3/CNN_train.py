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

def processData(Data):
    Result = []
    Label = []
    for i in range(10):
        for j in range(500):
            Datum = Data[i][j]
            Result.append([unFlatten(e) for e in
                           [Datum[:1024], Datum[1024:2048], Datum[2048:3072]]
                           ])
            Label.append([1 if e == i else 0 for e in range(10)])
    return np.array(Result), np.array(Label)

def processUnlabel(Data):
    Result = []
    for i in range(45000):
        Datum = Data[i]
        Result.append([unFlatten(e) for e in
                        [Datum[:1024], Datum[1024:2048], Datum[2048:3072]]
                        ])
    return np.array(Result)

def main():
    if len(argv) != 4:
        die('Usage: {} [labeled data] [unlabeled data] [model]'
            .format(argv[0]))

    X, Y = processData(pickle.load(open(argv[1], 'rb')))
    X_ = processUnlabel(pickle.load(open(argv[2], 'rb')))

    set_image_dim_ordering('th')
    Model = Sequential()
    Model.add(Convolution2D(25, 3, 3, input_shape = (3, 32, 32)))
    Model.add(MaxPooling2D((2, 2)))
    Model.add(Convolution2D(50, 3, 3))
    Model.add(MaxPooling2D((2, 2)))
    Model.add(Convolution2D(100, 3, 3))
    Model.add(Flatten())
    # Dim = 1600
    Model.add(Dense(output_dim = 400))
    Model.add(Activation('relu'))
    Model.add(Dense(output_dim = 100))
    Model.add(Activation('relu'))
    Model.add(Dense(output_dim = 25))
    Model.add(Activation('relu'))
    Model.add(Dense(output_dim = 10))
    Model.add(Activation('softmax'))

    Opt = SGD(lr = 1e-5)
    Model.compile(optimizer = Opt, loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    Model.fit(X, Y, verbose = 2, batch_size = 32, nb_epoch = 30)

    Sizes = [1000] * 20 + [5000] * 5
    for Size in Sizes:
        Y_ = Model.predict(X_, verbose = 1)
        print('\n')

        gather = zip(map((lambda e: e.max()), Y_), range(len(Y_)))
        gather = sorted(gather, reverse=True)
        gather = list(map((lambda e: e[1]), gather))[:Size]
        extendX = []
        extendY = []
        for i in gather:
            Label = 0
            for j in range(1, 10):
                if Y_[i][j] > Y_[i][Label]:
                    Label = j
            extendX.append(X_[i])
            extendY.append([1 if e == Label else 0 for e in range(10)])
        X = np.append(X, extendX, axis=0)
        Y = np.append(Y, extendY, axis=0)
        X_ = np.delete(X_, gather, axis=0)

        Model.fit(X, Y, verbose = 2, batch_size = 32, nb_epoch = 5)

    Model.fit(X, Y, verbose = 2, batch_size = 32, nb_epoch = 5)

    Model.save(argv[3])
    return 0

if __name__ == '__main__': main()
