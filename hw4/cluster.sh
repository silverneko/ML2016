#!/bin/sh

THEANO_FLAGS=device=gpu,floatX=float32 python3 cluster.py "$1/title_StackOverflow.txt" "$1/check_index.csv" $2
