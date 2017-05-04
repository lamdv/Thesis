import numpy as np
import re
import matplotlib.pyplot as plot

with  open('House_Price.txt', 'r') as input:
    A = np.array([[0, 0]])
    y = np.array([0])
    input.readline()
    for line in input.readlines():
        wordList = re.sub("[^\w]", " ", line).split()
        A = np.append(A, [[1, wordList[0]]], axis=0)
        y = np.append(y, [wordList[1]], axis=0)

w = np.linalg.lstsq(A, y)[0]
print('Least square weight: ')
print(w)
print(A.dtype())
fit_fn = np.poly1d(w) 
plot.plot(A[:,1],y,'ro',fit_fn,'--k')
plot.show()
