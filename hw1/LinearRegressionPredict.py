from sys import argv

def Transpose(A):
    B = [[] for _ in range(len(A[0]))]
    for row in A:
        for i in range(len(row)):
            B[i].append(row[i])
    return B

def F(B, C, X):
    R = B
    for i in range(len(X)):
        for j in range(N):
            R += C[i][j] * X[i][j]
    return R

CoefDataFile = open(argv[1], "r")
TestDataFile = open(argv[2], "r")
OutputFile = open("linear_regression.csv", "w")

N = 18

B = 0.0
C = [[0.0 for _ in range(N)] for _ in range(9)]
Input = []
for line in CoefDataFile:
    Input.append([float(s) for s in line.rstrip("\r\n").split(",")])
B = Input[0][0]
for i in range(9):
    for j in range(N):
        C[i][j] = Input[i+1][j]

OutputFile.write("id,value\n")
while True:
    X = []
    for i in range(N):
        line = TestDataFile.readline()
        if not line:
            exit()
        row = line.rstrip("\r\n").split(",")
        X.append(row[2:])
    Id = row[0]

    for i in range(len(X[10])):
        if X[10][i] == "NR":
            X[10][i] = 0

    for row in X:
        for i in range(len(row)):
            row[i] = float(row[i])

    X = Transpose(X)
    Y = F(B, C, X)
    OutputFile.write("%s,%f\n" % (Id, Y))


CoefDataFile.close()
TestDataFile.close()
OutputFile.close()
