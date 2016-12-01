#!/bin/sh

THEANO_FLAGS=device=gpu,floatX=float32 python3 tf_idf.py $1 $2
