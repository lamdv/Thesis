import numpy as np
import re

#network parameters
width = 20
func = np.random.rand(width, 1) #randomly assign activation function to the nodes of the network

#dictionary
genders = {'male':0, 'female':1}
ranks = {"full":0, "associate":1, "assistant":2}
degrees = {"masters":0, "doctorate":1}

#data input from ../salary.dat
#convert text input to numeric according to dictionary above
def inp(file):
    with  open(file, 'r') as input:
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
    return np.delete(A, 0, axis = 0), np.delete(y, 0), length

#feed forward into the network
def feed_forward(A, syn0, width):
    l0 = A
    l1 = np.array([sigmoid(np.dot(l0, syn0[:,1]))]) 
    for iter_width in range(width-1):
        if(func[iter_width] < 0.5):
            l1 = np.append(l1, [linear(np.dot(l0, syn0[:,iter_width]))],axis = 0) # Linear activation function nodes
        else:
            l1 = np.append(l1, [sigmoid(np.dot(l0, syn0[:,iter_width]))],axis = 0) # Sigmoid activation function nodes
    return l1

#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#linear function
def linear(x):
    return x

#read the training dataset
[A, y, length] = inp('salary.dat')
#randomized input weights
syn0 = 2*np.random.random((6,width)) - 1
out = feed_forward(A, syn0, width)
w = np.linalg.lstsq(out.T, y)[0] #least square learning on the output weight of random layer


#read the test dataset
[A, y, length] = inp("salary_test.dat")
#feed test data into network
out = feed_forward(A, syn0, width)
#calculate error
print(np.average(np.dot(out.T, w) - y))