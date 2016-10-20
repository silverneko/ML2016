#!/usr/bin/env python3

import numpy as np
import math
from sys import argv, stderr

def die(msg):
    print(msg, file = stderr)
    exit(1)

def randomInitWeight():
    w = np.random.random(57) * 2 - 1
    b = np.random.random() * 2 - 1
    w *= 0.001
    b *= 0.001
    return w, b

def sigmoid(z):
    if z > 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        z = math.exp(z)
        return z / (1.0 + z)

def f(w, b, x):
    return sigmoid((w * x).sum() + b)

def loss(w, b, Data):
    res = 0.0
    for (x, y) in Data:
        fwb = f(w, b, x)
        PredictY = 1 if f(w, b, x) > 0.5 else 0
        res += abs(PredictY - y)
    return res

def gradient(w, b, x, y):
    y_ = f(w, b, x) - y
    return y_ * x, y_

def main():
    if len(argv) != 3:
        die('Usage {} [train data] [output model]'.format(argv[0]))

    Data = []
    for line in open(argv[1], 'r'):
        row = line.rstrip('\r\n').split(',')
        for i in range(1, len(row)):
            row[i] = float(row[i])
        Data.append((np.array(row[1:-1]), row[-1]))

    MaxIterations = 10000
    LR = 0.01
    w, b = randomInitWeight()
    AccuGw = np.zeros(w.shape)
    AccuGb = 0.0
    for i in range(MaxIterations):
        print (loss(w, b, Data))
        gw = np.zeros(w.shape)
        gb = 0.0
        for (x, y) in Data:
            g = gradient(w, b, x, y)
            gw += g[0]
            gb += g[1]

        AccuGw += gw ** 2
        AccuGb += gb ** 2
        w -= LR * gw / np.sqrt(AccuGw)
        b -= LR * gb / np.sqrt(AccuGb)

    ModelFD = open(argv[2], 'w')
    ModelFD.write(' '.join([str(e) for e in w]))
    ModelFD.write('\n{}\n'.format(b))
    ModelFD.close()

if __name__ == '__main__':
    main()
