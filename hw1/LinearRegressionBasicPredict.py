from sys import argv

def F(B, C, X):
    R = B
    for i in range(len(X)):
        R += C[i] * X[i]
    return R

CoefDataFile = open(argv[1], "r")
TestDataFile = open(argv[2], "r")
OutputFile = open("linear_regression.csv", "w")

B = 0.0
C = [0.0] * 9
Input = []
for line in CoefDataFile:
    Input.append([float(s) for s in line.rstrip("\r\n").split(",")])
B = Input[0][0]
for i in range(9):
    C[i] = Input[1][i]

OutputFile.write("id,value\n")
while True:
    X = []
    for i in range(18):
        line = TestDataFile.readline()
        if not line:
            exit()
        row = line.rstrip("\r\n").split(",")
        X.append(row[2:])
    Id = row[0]

    X = X[9]

    for i in range(len(X)):
        X[i] = float(X[i])

    Y = F(B, C, X)
    OutputFile.write("%s,%f\n" % (Id, Y))


CoefDataFile.close()
TestDataFile.close()
OutputFile.close()
