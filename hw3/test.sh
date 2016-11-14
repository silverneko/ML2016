#!/bin/sh

THEANO_FLAGS=device=gpu,floatX=float32 python3 CNN_test.py "$1/test.p" $2 $3
