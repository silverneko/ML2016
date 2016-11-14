#!/usr/bin/env python3

from sys import argv, stderr
import pickle
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.backend import set_image_dim_ordering
from keras.optimizers import SGD, Adagrad, RMSprop
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

def main():
    if len(argv) != 4:
        die('Usage: {} [training data] [testing data] [prediction]'
            .format(argv[0]))

    set_image_dim_ordering('th')

    X, Y = processData(pickle.load(open(argv[1], 'rb')))
    TestID, TestData = processTest(pickle.load(open(argv[2], 'rb')))
    X = X / 255.0
    TestData = TestData / 255.0

    Model = Sequential()
    populateModel(Model)
    Model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    Datagen = ImageDataGenerator(horizontal_flip = True,
                                 width_shift_range = 0.1,
                                 height_shift_range = 0.1,
                                 rotation_range = 10)
    Model.fit_generator(Datagen.flow(X, Y, batch_size = 32),
                        samples_per_epoch = X.shape[0],
                        nb_epoch = 100)
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
