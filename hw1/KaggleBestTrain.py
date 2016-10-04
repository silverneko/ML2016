from sys import argv
import math
import numpy as np

def loadTrainingData(Filename):
    TrainDataFile = open(Filename, "r")
    lines = TrainDataFile.readlines()
    TrainDataFile.close()

    X = [[] for _ in range(18)]
    for i in range(1, len(lines), 18):
        for j in range(18):
            row = lines[i+j].rstrip("\r\n").split(",")
            X[j].extend(row[3:])

    for i in range(len(X[10])):
        if X[10][i] == "NR":
            X[10][i] = 0.0

    for row in X:
        for i in range(len(row)):
            row[i] = float(row[i])

    return np.array(X).T

def loadCoefficient(Filename):
    File = open(Filename, "r")
    lines = File.readlines()
    File.close()
    B = float(lines[0].rstrip("\r\n"))
    Input = []
    for i in range(1, len(lines)):
        Input.append([float(s) for s in lines[i].rstrip("\r\n").split(",")])
    C = np.array(Input)
    return B, C

def saveCoefficient(Filename, B, C):
    File = open(Filename, "w")
    File.write(str(B))
    File.write("\n")
    for row in C:
        File.write(",".join([str(e) for e in row]))
        File.write("\n")
    File.close()

def F(B, C, X):
    return (C * X).sum() + B

def L(B, C, X):
    W = 0.0
    for i in range(9, len(X)):
        Y = F(B, C, X[i-9:i])
        W += (X[i, 9] - Y) ** 2
    return W

def main():
    X = loadTrainingData(argv[1])
    B, C = loadCoefficient(argv[2])
    if len(argv) > 3:
        Iterations = int(argv[3])
    else:
        Iterations = 10000

    Alpha = 0.01
    AccuGradB = 0.0
    AccuGradC = np.zeros((9, 18))
    for _ in range(Iterations):
        if _ % 100 == 0:
            print (L(B, C, X) / (len(X) - 9))
            saveCoefficient("coefficient_best.csv", B, C)

        GradB = 0.0
        GradC = np.zeros((9, 18))
        for i in range(9, len(X)):
            Y = F(B, C, X[i-9:i])
            GradB += (Y - X[i, 9])
            GradC += (Y - X[i, 9]) * X[i-9:i]
        GradB /= (len(X) - 9)
        GradC /= (len(X) - 9)
        AccuGradB += GradB ** 2
        AccuGradC += GradC ** 2
        B -= Alpha * GradB / math.sqrt(AccuGradB)
        C -= Alpha * GradC / np.sqrt(AccuGradC)

    saveCoefficient("coefficient_best.csv", B, C)

if __name__ == "__main__":
    main()
