import numpy as np
import re

width = 10
genders = {'male':0, 'female':1}
ranks = {"full":0, "associate":1, "assistant":2}
degrees = {"masters":0, "doctorate":1}

with  open('salary.dat', 'r') as input:
    length = 0
    A = np.array([[0, 0, 0, 0, 0, 0]]) #input layer size of 6
    y = np.array([0])
    for line in input.readlines():
        wordList = re.sub("[^\w]", " ", line).split()
        gender = genders[wordList[0]]
        rank = ranks[wordList[1]]
        degree = degrees[wordList[3]]
        vect = np.array([1, float(genders[wordList[0]]), ranks[wordList[1]], float(wordList[2]), degrees[wordList[3]], float(wordList[4])]) #gender, rank, yr, degree received, yd
        A = np.append(A, [vect], axis=0)
        y = np.append(y, [float(wordList[5])], axis=0)
        length += 1

#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#sigmoid derivative function
def deriv(x):
    return x*(1-x)

np.random.seed()

syn0 = 2*np.random.random((6,width)) - 1

#for iter in range(10000):
l0 = A #input layer
#feed forward into the network
l1 = np.array([sigmoid(np.dot(l0, syn0[:,1]))]) 
for iter_width in range(width-1):
    l1 = np.append(l1, [sigmoid(np.dot(l0, syn0[:,iter_width]))],axis = 0) # Single (width) wide feed forward perceptron with randomized bias

#print (syn0)
#print (l1.T)
#print(A, y)
w = np.linalg.lstsq(l1.T, y)[0] #least square learning on the output weight of random layer

err = np.array([0])
for i in range(0, length-1, 1):
    err = np.append(err, np.dot(l1.T[i,:], w) - y[i])

print("Error: ", np.average(err))