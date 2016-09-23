from sys import argv

colId = int(argv[1])
filename = argv[2]

file = open(filename, "r")
outputFile = open("ans1.txt", "w")

list = []
for line in file:
    words = filter(lambda x: len(x) > 0, line.split(" "))
    number = float(words[colId])
    list.append(number)

list.sort()
outputFile.write(",".join([str(f) for f in list]))
