#!/bin/sh

THEANO_FLAGS=device=gpu,floatX=float32 python3 CNN_Supervised.py "$1/all_label.p" "$1/test.p" $2
