#!/bin/sh

THEANO_FLAGS=device=gpu,floatX=float32 python3 CNN_train.py "$1/all_label.p" "$1/all_unlabel.p" $2
