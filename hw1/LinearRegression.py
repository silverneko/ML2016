from sys import argv
import math
import numpy as np
from collections import deque

def loadTrainingData(Filename):
    TrainDataFile = open(Filename, "r")
    lines = TrainDataFile.readlines()
    TrainDataFile.close()

    data = []
    for m in range(1, len(lines), 18 * 20):
        X = [[] for _ in range(18)]
        for d in range(0, 18 * 20, 18):
            for i in range(18):
                row = lines[m + d + i].rstrip('\r\n').split(',')
                X[i].extend(row[3:])

        for i in range(len(X[10])):
            if X[10][i] == "NR":
                X[10][i] = 0.0

        for row in X:
            for i in range(len(row)):
                row[i] = float(row[i])

        d = deque()
        for i in range(9):
            d.append([X[j][i] for j in range(18)])
        for i in range(9, 20 * 24):
            data.append((np.array(d), X[9][i]))
            d.popleft()
            d.append([X[j][i] for j in range(18)])

    return data

def loadCoefficient(Filename):
    File = open(Filename, "r")
    lines = File.readlines()
    File.close()
    B = float(lines[0].rstrip("\r\n"))
    Input = []
    for i in range(1, 10):
        Input.append([float(s) for s in lines[i].rstrip("\r\n").split(",")])
    C = np.array(Input)
    return B, C

def loadTestData(Filename):
    File = open(Filename, "r")
    lines = File.readlines()
    File.close()
    Input = []
    for i in range(0, len(lines), 18):
        X = []
        for j in range(18):
            row = lines[i+j].rstrip("\r\n").split(",")
            X.append(row[2:])
        Id = row[0]

        for j in range(len(X[10])):
            if X[10][j] == "NR":
                X[10][j] = 0

        for row in X:
            for j in range(len(row)):
                row[j] = float(row[j])

        Input.append((Id, np.array(X).T))

    return Input

def F(B, C, X):
    return (C * X).sum() + B

def Loss(B, C, X):
    W = 0.0
    RegularizeTerm = 0.0
    for x, y in X:
        Y = F(B, C, x)
        W += (y - Y) ** 2
    return W / len(X)

def Gradient(B, C, x, y):
    Y = F(B, C, x)
    return (Y - y), ((Y - y) * x)

def main():
    X = loadTrainingData(argv[1])
    B, C = loadCoefficient(argv[2])
    B = np.random.random() - 0.5
    C = np.random.random(C.shape) - 0.5
    TestSets = loadTestData(argv[3])

    Alpha = 10
    AccuGradB = 1e-20
    AccuGradC = np.full(C.shape, 1e-20)
    for Iteration in range(20000):
        if Iteration % 100 == 0:
            print (Iteration, Loss(B, C, X))
        GradB = 0.0
        GradC = np.zeros((9, 18))
        for x, y in X:
            g = Gradient(B, C, x, y)
            GradB += g[0]
            GradC += g[1]
        GradB /= len(X)
        GradC /= len(X)
        AccuGradB += GradB ** 2
        AccuGradC += GradC ** 2
        B -= Alpha * GradB / math.sqrt(AccuGradB)
        C -= Alpha * GradC / np.sqrt(AccuGradC)

    print (Iteration+1, Loss(B, C, X))

    if len(argv) > 4:
        Filename = argv[4]
    else:
        Filename = "linear_regression.csv"
    OutputFile = open(Filename, "w")
    OutputFile.write("id,value\n")
    for (Id, X) in TestSets:
        Y = F(B, C, X)
        OutputFile.write("%s,%f\n" % (Id, Y))
    OutputFile.close()

if __name__ == "__main__":
    main()
