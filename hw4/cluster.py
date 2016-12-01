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
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

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
    if len(argv) != 4:
        die('Usage: {} [document file] [check file] [output result]'
            .format(argv[0]))

    Docs = open(argv[1], 'r').readlines()
    for i in range(len(Docs)):
        Docs[i] = unicodedata.normalize('NFKC', Docs[i])

    #vectorizer = joblib.load('vectorizer.pkl')
    #km = joblib.load('kmeans.pkl')
    vectorizer = sktext.TfidfVectorizer(stop_words='english',
                                        max_df=0.5,
                                        min_df=2,
                                        tokenizer=MyTokenizer())
    tfidf = vectorizer.fit_transform(Docs)
    tfidf = tfidf.toarray()

    Words = vectorizer.get_feature_names()
    n_features = len(Words)
    print ('Number of Features: ', n_features)

    n_clusters = 80

    km = skcluster.MiniBatchKMeans(n_clusters=n_clusters,
                                   max_iter=300,
                                   n_init=20,
                                   batch_size=1000,
                                   verbose=1)

    X = tfidf
    svd = TruncatedSVD(500)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)

    Y = km.fit_predict(X)

    Queries = open(argv[2], 'r').readlines()[1:]
    Result = open(argv[3], 'w')
    Result.write('ID,Ans\n')
    for query in Queries:
        [id, xid, yid] = [int(e) for e in query.split(',')]
        if Y[xid] == Y[yid]:
            Result.write('{},{}\n'.format(id, 1))
        else:
            Result.write('{},{}\n'.format(id, 0))

    Result.close()

    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(km, 'kmeans.pkl')

    for i in range(100):
        [id, xid, yid] = [int(e) for e in Queries[i].split(',')]
        print (Docs[xid], Docs[yid], Y[xid] == Y[yid])

    lsa2 = TruncatedSVD(n_components=2)
    X = lsa2.fit_transform(tfidf)
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, n_clusters))
    plt.figure()
    for i in range(n_clusters):
        plt.scatter(X[Y == i, 0], X[Y == i, 1], color=colors[i], alpha=0.6)
    plt.show()



if __name__ == '__main__': main()
