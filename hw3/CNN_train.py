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
from keras.layers import Input, UpSampling2D
from keras.backend import set_image_dim_ordering
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
from sklearn import svm

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
            Label.append(i)
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

    Datagen = ImageDataGenerator(horizontal_flip = True,
                                 width_shift_range = 0.1,
                                 height_shift_range = 0.1,
                                 rotation_range = 10)

    input_img = Input(shape = (3, 32, 32))
    x = Convolution2D(16, 3, 3, border_mode = 'same')(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, border_mode = 'same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, border_mode = 'same')(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2))(x)
    encoded_feature = Flatten()(encoded)

    x = Convolution2D(8, 3, 3, border_mode = 'same')(encoded)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, border_mode = 'same')(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 3, 3, border_mode = 'same')(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(3, 3, 3, border_mode = 'same')(x)
    decoded = Activation('relu')(x)

    Autoencoder = keras.models.Model(input_img, decoded)
    Autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

    X_all = np.append(X, X_, axis = 0)
    def Gen():
        for x in Datagen.flow(X_all, batch_size = 32):
            yield (x, x)

    Autoencoder.fit_generator(Gen(),
                              samples_per_epoch = X_all.shape[0],
                              nb_epoch = 50)

    """
    import matplotlib.pyplot as plt
    x_test = X[:10]
    decoded_imgs = Autoencoder.predict(x_test)
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i][0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i][0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    """

    Encoder = keras.models.Model(input_img, encoded_feature)

    # Make validation set
    if MakeValidationSet:
        XY = list(zip(X, Y))
        random.shuffle(XY)
        X, Y = zip(*XY[:4500])
        X_test, Y_test = zip(*XY[4500:])
        X = np.array(X)
        Y = np.array(Y)
        X_test = np.array(X_test)
        Y_test = [[1 if e == i else 0 for i in range(10)] for e in Y_test]
        Y_test = np.array(Y_test)

    X_encode = Encoder.predict(X)
    clf = svm.SVC()
    clf.fit(X_encode, Y)
    X_unlabel_encode = Encoder.predict(X_)
    Y_ = clf.predict(X_unlabel_encode)

    Y = [[1 if e == i else 0 for i in range(10)] for e in Y]
    Y_ = [[1 if e == i else 0 for i in range(10)] for e in Y_]
    Y = np.array(Y)
    Y_ = np.array(Y_)

    X = np.append(X, X_, axis = 0)
    Y = np.append(Y, Y_, axis = 0)

    Model = Sequential()
    populateModel(Model)
    Model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    if MakeValidationSet:
        Model.fit_generator(Datagen.flow(X, Y, batch_size = 32),
                            samples_per_epoch = X.shape[0],
                            nb_epoch = 100,
                            validation_data = (X_test, Y_test))
    else:
        Model.fit_generator(Datagen.flow(X, Y, batch_size = 32),
                            samples_per_epoch = X.shape[0],
                            nb_epoch = 100)


    Model.save(argv[3])
    return 0

if __name__ == '__main__': main()
