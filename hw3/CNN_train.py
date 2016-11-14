#!/usr/bin/env python3

from sys import argv, stderr
import pickle
import numpy as np
import random
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.backend import set_image_dim_ordering
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator

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

def populateModel(Model):
    Model.add(Convolution2D(32, 3, 3, input_shape = (3, 32, 32)))
    Model.add(Activation('relu'))
    Model.add(Convolution2D(32, 3, 3))
    Model.add(Activation('relu'))
    Model.add(MaxPooling2D((2, 2)))
    Model.add(Dropout(0.25))
    Model.add(Convolution2D(64, 3, 3))
    Model.add(Activation('relu'))
    Model.add(MaxPooling2D((2, 2)))
    Model.add(Dropout(0.25))
    Model.add(Flatten())
    Model.add(Dense(output_dim = 256))
    Model.add(Activation('relu'))
    Model.add(Dense(output_dim = 256))
    Model.add(Activation('relu'))
    Model.add(Dropout(0.25))
    Model.add(Dense(output_dim = 10))
    Model.add(Activation('softmax'))

def entropy(P):
    result = 0.0
    for p in P:
        if p > 0.0:
            result += p * math.log10(p)
    result *= -1
    return result

def main():
    MakeValidationSet = False

    if len(argv) != 4:
        die('Usage: {} [labeled data] [unlabeled data] [model]'
            .format(argv[0]))

    set_image_dim_ordering('th')

    X, Y = processData(pickle.load(open(argv[1], 'rb')))
    X_ = processUnlabel(pickle.load(open(argv[2], 'rb')))
    X = X / 255.0
    X_ = X_ / 255.0

    # Make validation set
    if MakeValidationSet:
        XY = list(zip(X, Y))
        random.shuffle(XY)
        X, Y = zip(*XY[:4500])
        X_test, Y_test = zip(*XY[4500:])
        X = np.array(X)
        Y = np.array(Y)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

    Model = Sequential()
    populateModel(Model)
    Model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    Datagen = ImageDataGenerator(horizontal_flip = True,
                                 width_shift_range = 0.1,
                                 height_shift_range = 0.1,
                                 rotation_range = 10)
    if MakeValidationSet:
        Model.fit_generator(Datagen.flow(X, Y, batch_size = 32),
                            samples_per_epoch = X.shape[0],
                            nb_epoch = 60,
                            validation_data = (X_test, Y_test))
    else:
        Model.fit_generator(Datagen.flow(X, Y, batch_size = 32),
                            samples_per_epoch = X.shape[0],
                            nb_epoch = 60)

    Y_ = Model.predict(X_, verbose = 1)
    print('')

    pv = enumerate([entropy(e) for e in Y_])
    gather = [e[0] for e in filter((lambda e: e[1] < 0.1), pv)]

    extendX = []
    extendY = []
    for i in gather:
        Label = 0
        for j in range(1, 10):
            if Y_[i][j] > Y_[i][Label]:
                Label = j
        extendX.append(X_[i])
        extendY.append([1 if e == Label else 0 for e in range(10)])
    X = np.append(X, extendX, axis = 0)
    Y = np.append(Y, extendY, axis = 0)

    Model2 = Sequential()
    populateModel(Model2)
    Model2.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    if MakeValidationSet:
        Model2.fit_generator(Datagen.flow(X, Y, batch_size = 32),
                                samples_per_epoch = X.shape[0],
                                nb_epoch = 100,
                                validation_data = (X_test, Y_test))
    else:
        Model2.fit_generator(Datagen.flow(X, Y, batch_size = 32),
                                samples_per_epoch = X.shape[0],
                                nb_epoch = 100)

    Model2.save(argv[3])
    return 0

if __name__ == '__main__': main()
