#!/usr/bin/env python3

import numpy as np
import math
from sys import argv, stderr

def die(msg):
    print(msg, file = stderr)
    exit(1)

def main():
    if len(argv) != 3:
        die('Usage {} [train data] [output model]'.format(argv[0]))

    Data = [[],[]]
    for line in open(argv[1], 'r'):
        row = line.rstrip('\r\n').split(',')
        for i in range(1, len(row)-1):
            row[i] = float(row[i])
        label = int(row[-1])
        Data[label].append(np.array(row[1:-1]))

    uv = [np.zeros(57) for i in range(2)]
    for i in range(2):
        for y in Data[i]:
            uv[i] += y
        uv[i] /= len(Data[i])

    sv = [np.zeros((57, 57)) for i in range(2)]
    for i in range(2):
        for y in Data[i]:
            d = y - uv[i]
            sv[i] += np.outer(d, d)
        sv[i] /= len(Data[i])

    ss = (len(Data[0]) * sv[0] + len(Data[1]) * sv[1]) / (len(Data[0]) + len(Data[1]))


    ModelFD = open(argv[2], 'w')
    ModelFD.write('{} {}\n'.format(len(Data[0]), len(Data[1])))
    ModelFD.write(' '.join([str(e) for e in uv[0]]))
    ModelFD.write('\n')
    ModelFD.write(' '.join([str(e) for e in uv[1]]))
    ModelFD.write('\n')
    for row in ss:
        ModelFD.write(' '.join([str(e) for e in row]))
        ModelFD.write('\n')
    ModelFD.close()

if __name__ == '__main__':
    main()
