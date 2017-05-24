import numpy as np
import re
import matplotlib.pyplot as plot

with  open('House_Price.txt', 'r') as input:
    A = np.array([[0, 0]])
    y = np.array([0])
    input.readline()
    for line in input.readlines():
        wordList = re.sub("[^\w]", " ", line).split()
        A = np.append(A, [[1, float(wordList[0])]], axis=0)
        y = np.append(y, [float(wordList[1])], axis=0)

w = np.linalg.lstsq(A, y)[0]
print('Least square weight: ')
print(w)


err = np.average(np.dot(A, w) - y)

print("Error: ", np.average(err))