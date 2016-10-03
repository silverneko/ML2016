from sys import argv
import math

CoefDataFile = open(argv[1], "r")
TrainDataFile = open(argv[2], "r")
OutputCoef = open("coefficient_best.csv", "w")

TrainData = []
for line in TrainDataFile:
    TrainData.append(line.rstrip("\r\n").split(","))
TrainData.pop(0)

N = 18
X = [[] for _ in range(N)]
for i in range(len(TrainData) / N):
    for j in range(N):
        index = i * N + j
        X[j].extend(TrainData[index][3:])

for i in range(len(X[10])):
    if X[10][i] == "NR":
        X[10][i] = 0

for row in X:
    for i in range(len(row)):
        row[i] = float(row[i])

def Transpose(A):
    B = [[] for _ in range(len(A[0]))]
    for row in A:
        for i in range(len(row)):
            B[i].append(row[i])
    return B

X = Transpose(X)

def F(B, C, X):
    R = B
    for i in range(len(X)):
        for j in range(N):
            R += C[i][j] * X[i][j]
    return R

def L(B, C, X):
    W = 0.0
    for i in range(9, len(X)):
        Y = F(B, C, X[i-9:i])
        W += (X[i][9] - Y) ** 2
    return W

B = 0.0
C = [[0.0 for _ in range(N)] for _ in range(9)]
Input = []
for line in CoefDataFile:
    Input.append([float(s) for s in line.rstrip("\r\n").split(",")])
B = Input[0][0]
for i in range(9):
    for j in range(N):
        C[i][j] = Input[i+1][j]

Alpha = 0.01
AccuGradB = 0.0
AccuGradC = [[0.0 for _ in range(N)] for _ in range(9)]
Iterations = 10000
for _ in range(Iterations):
    if _ % 25 == 0:
        OutputCoef.seek(0)
        OutputCoef.truncate()
        OutputCoef.write(str(B))
        OutputCoef.write("\n")
        for row in C:
            OutputCoef.write(",".join([str(e) for e in row]))
            OutputCoef.write("\n")
        OutputCoef.flush()

    # print (L(B, C, X) / (len(X) - 9))
    W = 0.0

    GradB = 0.0
    GradC = [[0.0 for _ in range(N)] for _ in range(9)]
    for i in range(9, len(X)):
        Y = F(B, C, X[i-9:i])
        W += (X[i][9] - Y) ** 2
        GradB += (Y - X[i][9])
        for j in range(9):
            for k in range(N):
                GradC[j][k] += (Y - X[i][9]) * X[i-9+j][k]

    print (W / (len(X) - 9))

    GradB /= (len(X) - 9)
    AccuGradB += GradB ** 2
    for i in range(9):
        for j in range(N):
            GradC[i][j] /= (len(X) - 9)
            AccuGradC[i][j] += GradC[i][j] ** 2

    B -= Alpha * GradB / math.sqrt(AccuGradB)
    for i in range(9):
        for j in range(N):
            C[i][j] -= Alpha * GradC[i][j] / math.sqrt(AccuGradC[i][j])

OutputCoef.seek(0)
OutputCoef.truncate()
OutputCoef.write(str(B))
OutputCoef.write("\n")
for row in C:
    OutputCoef.write(",".join([str(e) for e in row]))
    OutputCoef.write("\n")

CoefDataFile.close()
TrainDataFile.close()
OutputCoef.close()
