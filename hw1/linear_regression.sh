#!/usr/bin/env sh

python ./LinearRegressionTrain.py ./data/train.csv ./coef_zero.csv 1000 > /dev/null
python ./LinearRegressionPredict.py ./data/test_X.csv ./coefficient.csv linear_regression.csv
