from sys import argv
import math
import numpy as np

def loadCoefficient(Filename):
    File = open(Filename, "r")
    lines = File.readlines()
    File.close()
    B = float(lines[0].rstrip("\r\n"))
    Input = []
    for i in range(1, 10):
        Input.append([float(s) for s in lines[i].rstrip("\r\n").split(",")])
    C = np.array(Input)
    Input = []
    for i in range(10, 19):
        Input.append([float(s) for s in lines[i].rstrip("\r\n").split(",")])
    normalized_attr = (np.array(Input),)
    Input = []
    for i in range(19, 28):
        Input.append([float(s) for s in lines[i].rstrip("\r\n").split(",")])
    normalized_attr = normalized_attr + (np.array(Input),)
    normalized_attr += tuple([float(s) for s in lines[28].rstrip("\r\n").split(",")])

    return B, C, normalized_attr

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

def Normalize(X, normalized_attr):
    for i in range(len(X)):
        X[i] = X[i][0], (X[i][1] - normalized_attr[0]) / normalized_attr[1]

def F(B, C, X):
    return (C * X).sum() + B

def main():
    if len(argv) > 3:
        OutputFile = open(argv[3], "w")
    else:
        OutputFile = open("linear_regression.csv", "w")
    OutputFile.write("id,value\n")

    TestSets = loadTestData(argv[1])
    B, C, normalized_attr = loadCoefficient(argv[2])
    Normalize(TestSets, normalized_attr)
    for (Id, X) in TestSets:
        Y = F(B, C, X)
        Y = Y * normalized_attr[3] + normalized_attr[2]
        OutputFile.write("%s,%f\n" % (Id, Y))

    OutputFile.close()

if __name__ == "__main__":
    main()
