import numpy as np
import re

genders = {'male':0, 'female':1}
ranks = {"full":0, "associate":1, "assistant":2}
degrees = {"masters":0, "doctorate":1}

with  open('salary.dat', 'r') as input:
    length = 0
    A = np.array([[0, 0, 0, 0, 0]])
    y = np.array([0])
    for line in input.readlines():
        wordList = re.sub("[^\w]", " ", line).split()
        gender = genders[wordList[0]]
        rank = ranks[wordList[1]]
        degree = degrees[wordList[3]]
        vect = np.array([genders[wordList[0]], ranks[wordList[1]], wordList[2], degrees[wordList[3]], wordList[4]]) #gender, rank, yr, degree received, yd
        A = np.append(A, [vect], axis=0)
        y = np.append(y, [wordList[5]], axis=0)
        length += 1
w = np.linalg.lstsq(A, y)[0]
print('Least square weight: ')
print(w)
