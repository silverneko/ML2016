#!/usr/bin/env sh

python ./KaggleBestTrain.py ./data/train.csv ./coef_zero.csv 1000 > /dev/null
python ./LinearRegressionPredict.py ./data/test_X.csv ./coefficient_best.csv kaggle_best.csv
