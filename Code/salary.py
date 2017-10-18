import numpy as np
import re

genders = {'male':0, 'female':1}
ranks = {"full":0, "associate":1, "assistant":2}
degrees = {"masters":0, "doctorate":1}

#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def inp(file):
    with  open(file, 'r') as input:
        length = 0
        A = np.array([[0, 0, 0, 0, 0, 0]]) #input layer size of 6
        y = np.array([0])
        for line in input.readlines():
            wordList = re.sub("[^\w]", " ", line).split()
            vect = np.array([1, float(genders[wordList[0]]), ranks[wordList[1]], float(wordList[2]), degrees[wordList[3]], float(wordList[4])]) #gender, rank, yr, degree received, yd
            A = np.append(A, [vect], axis=0)
            y = np.append(y, [float(wordList[5])], axis=0)
            length += 1
    return np.delete(A, 0, axis = 0), np.delete(y, 0), length
    # return np.delete(A, 0, axis = 0), np.delete(y, 0), length

[A, y, length] = inp('salary.dat')

A = sigmoid(A)

print(A)
        
w = np.linalg.lstsq(A, y)[0]
print('Least square weight: ')
print(w) 

[A, y, length] = inp('salary_test.dat')

A = sigmoid(A)

print ("result: ",np.dot(A,w))
err = np.average(np.dot(A, w) - y)

print("Error: ", err)