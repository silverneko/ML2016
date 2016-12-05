#!/usr/bin/env python3

from sys import argv, stderr
import pickle
import numpy as np
import random
import math
import unicodedata
import matplotlib.pyplot as plt
import sklearn.feature_extraction.text as sktext
import sklearn.cluster as skcluster
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import nltk
import string

def die(msg):
    print(msg, file = stderr)
    exit(1)

class MyTokenizer(object):
    def __init__(self):
        self.stemmer = nltk.stem.porter.PorterStemmer()

    def __call__(self, doc):
        doc = ''.join([ch if ch not in string.punctuation else ' ' for ch in doc])
        return [self.stemmer.stem(tk) for tk in nltk.word_tokenize(doc) if len(tk) >= 2]


def main():
    if len(argv) != 2:
        die('Usage: {} [document file]'
            .format(argv[0]))

    Docs = open(argv[1], 'r').readlines()
    for i in range(len(Docs)):
        Docs[i] = unicodedata.normalize('NFKC', Docs[i])

    vectorizer = sktext.TfidfVectorizer(stop_words=None,
                                        max_df=0.05,
                                        min_df=1,
                                        tokenizer=MyTokenizer())

    tfidf = vectorizer.fit_transform(Docs)
    tfidf = tfidf.toarray()

    print (vectorizer.stop_words_)


if __name__ == '__main__': main()
