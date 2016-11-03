#!/bin/sh

python3 CNN_train.py "$1/all_label.p" "$1/all_unlabel.p" $2
