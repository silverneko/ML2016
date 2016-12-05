#!/usr/bin/env python3

from sys import argv, stderr
import pickle
import numpy as np
import random
import math
import unicodedata
import matplotlib.pyplot as plt
import sklearn
import nltk
import string

def die(msg):
    print(msg, file = stderr)
    exit(1)

def main():
    if len(argv) != 4:
        die('Usage: {} [check file] [label] [prediction]'
            .format(argv[0]))

    Queries = open(argv[1], 'r').readlines()[1:]
    Label = [int(e) for e in open(argv[2], 'r').readlines()]
    Predict = [int(e.split(',')[1]) for e in open(argv[3], 'r').readlines()[1:]]
    Label_Y = []

    for query in Queries:
        [id, xid, yid] = [int(e) for e in query.split(',')]
        if Label[xid] == Label[yid]:
            Label_Y.append(1)
        else:
            Label_Y.append(0)

    score = sklearn.metrics.fbeta_score(Label_Y, Predict, 0.25)
    print (score)


if __name__ == '__main__': main()
