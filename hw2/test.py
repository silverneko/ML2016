#!/usr/bin/env python3

import numpy as np
import math
from sys import argv, stderr

def die(msg):
    print(msg, file = stderr)
    exit(1)

def main():
    if len(argv) != 4:
        die('Usage {} [model] [test data] [output name]'.format(argv[0]))

    Model = open(argv[1], 'r').readlines()
    c = [float(e) for e in Model[0].split()]
    Model = Model[1:]
    uv = [[]] * 2
    uv[0] = np.array([float(e) for e in Model[0].split()])
    uv[1] = np.array([float(e) for e in Model[1].split()])
    Model = Model[2:]
    ss = [[]] * 57
    for i in range(57):
        ss[i] = [float(e) for e in Model[i].split()]
    ss = np.linalg.inv(np.array(ss))

    Output = open(argv[3], 'w')
    Output.write('id,label\n')
    for line in open(argv[2], 'r'):
        row = line.rstrip('\r\n').split(',')
        for i in range(1, len(row)):
            row[i] = float(row[i])
        Id = row[0]
        x = np.array(row[1:])

        p = [[]] * 2
        for i in range(2):
            p[i] = math.exp(-0.5 * np.dot(np.dot((x - uv[i]), ss), (x - uv[i])))
            p[i] = p[i] * c[i] / sum(c)

        PredictY = 1 if p[0] < p[1] else 0
        Output.write('{},{}\n'.format(Id, PredictY))

    Output.close()

if __name__ == '__main__':
    main()
