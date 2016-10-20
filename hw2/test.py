#!/usr/bin/env python3

import numpy as np
import math
from sys import argv, stderr

def die(msg):
    print(msg, file = stderr)
    exit(1)

def f(w, b, x):
    return (w * x).sum() + b

def main():
    if len(argv) != 4:
        die('Usage {} [model] [test data] [output name]'.format(argv[0]))

    Model = open(argv[1], 'r').readlines()
    w = np.array([float(e) for e in Model[0].split()])
    b = float(Model[1])

    Output = open(argv[3], 'w')
    Output.write('id,label\n')
    for line in open(argv[2], 'r'):
        row = line.rstrip('\r\n').split(',')
        for i in range(1, len(row)):
            row[i] = float(row[i])
        Id = row[0]
        PredictY = 1 if f(w, b, np.array(row[1:])) > 0.0 else 0
        Output.write('{},{}\n'.format(Id, PredictY))

    Output.close()

if __name__ == '__main__':
    main()
