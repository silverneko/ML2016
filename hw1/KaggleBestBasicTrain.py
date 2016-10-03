from sys import argv
import math

CoefDataFile = open(argv[1], "r")
TrainDataFile = open(argv[2], "r")
OutputCoef = open("coefficient_best_basic.csv", "w")

TrainData = []
for line in TrainDataFile:
    TrainData.append(line.rstrip("\r\n").split(","))
TrainData.pop(0)

X = []
for i in range(len(TrainData) / 18):
    index = i * 18 + 9
    X.extend(TrainData[index][3:])

for i in range(len(X)):
    X[i] = float(X[i])

def F(B, C, X):
    R = B
    for i in range(len(X)):
        R += C[i] * X[i]
    return R

def L(B, C, X):
    W = 0.0
    for i in range(9, len(X)):
        Y = F(B, C, X[i-9:i])
        W += (X[i] - Y) ** 2
    return W

B = 0.0
C = [0.0] * 9
Input = []
for line in CoefDataFile:
    Input.append([float(s) for s in line.rstrip("\r\n").split(",")])
B = Input[0][0]
for i in range(9):
    C[i] = Input[1][i]

Alpha = 0.001
AccuGradB = 0.0
AccuGradC = [0.0] * 9
Iterations = 10000
for _ in range(Iterations):
    if _ % 25 == 0:
        OutputCoef.seek(0)
        OutputCoef.truncate()
        OutputCoef.write(str(B))
        OutputCoef.write("\n")
        OutputCoef.write(",".join([str(e) for e in C]))
        OutputCoef.write("\n")
        OutputCoef.flush()

    # print (L(B, C, X) / (len(X) - 9))
    W = 0.0

    GradB = 0.0
    GradC = [0.0] * 9
    for i in range(9, len(X)):
        Y = F(B, C, X[i-9:i])
        W += (X[i] - Y) ** 2
        GradB += (Y - X[i])
        for j in range(9):
            GradC[j] += (Y - X[i]) * X[i-9+j]

    print (W / (len(X) - 9))

    GradB /= (len(X) - 9)
    AccuGradB += GradB ** 2
    for i in range(9):
        GradC[i] /= (len(X) - 9)
        AccuGradC[i] += GradC[i] ** 2

    B -= Alpha * GradB / math.sqrt(AccuGradB)
    for i in range(9):
        C[i] -= Alpha * GradC[i] / math.sqrt(AccuGradC[i])

OutputCoef.seek(0)
OutputCoef.truncate()
OutputCoef.write(str(B))
OutputCoef.write("\n")
OutputCoef.write(",".join([str(e) for e in C]))
OutputCoef.write("\n")

CoefDataFile.close()
TrainDataFile.close()
OutputCoef.close()
