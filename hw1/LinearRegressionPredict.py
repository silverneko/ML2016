from sys import argv
import math
import numpy as np

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

def main():
    if len(argv) > 3:
        OutputFile = open(argv[3], "w")
    else:
        OutputFile = open("linear_regression.csv", "w")
    OutputFile.write("id,value\n")

    TestSets = loadTestData(argv[1])
    B, C = loadCoefficient(argv[2])
    for (Id, X) in TestSets:
        Y = F(B, C, X)
        OutputFile.write("%s,%f\n" % (Id, Y))

    OutputFile.close()

if __name__ == "__main__":
    main()
