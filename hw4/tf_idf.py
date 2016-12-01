#!/usr/bin/env python3

from sys import argv, stderr
import pickle
import numpy as np
import random
import math
import sklearn.feature_extraction.text as sktext
import sklearn.cluster as skcluster
import unicodedata

def die(msg):
    print(msg, file = stderr)
    exit(1)

def main():
    if len(argv) != 3:
        die('Usage: {} [document file] [output vocabulary]'
            .format(argv[0]))

    Docs = open(argv[1], 'r').readlines()
    for i in range(len(Docs)):
        Docs[i] = unicodedata.normalize('NFKC', Docs[i])

    vectorizer = sktext.TfidfVectorizer(smooth_idf=True)
    tfidf = vectorizer.fit_transform(Docs)
    tfidf = tfidf.toarray()

    Words = vectorizer.get_feature_names()
    idfWords = sorted(zip(vectorizer.idf_, Words))
    print ([x[1] for x in idfWords[:50]])



if __name__ == '__main__': main()
