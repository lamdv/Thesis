import numpy as np
import re

np.random.seed()

#network parameters
width = 100
#func = np.random.rand(width, 1) #randomly assign activation function to the nodes of the network

#dictionary
genders = {'male':1, 'female':2}
ranks = {"full":1, "associate":2, "assistant":3}
degrees = {"masters":1, "doctorate":2}

#data input from ../salary.dat
#convert text input to numeric according to dictionary above
def inp(file):
    with  open(file, 'r') as input:
        length = 0
        A = np.array([[0, 0, 0, 0, 0, 0]]) #input layer size of 6
        y = np.array([0])
        for line in input.readlines():
            wordList = re.sub("[^\w]", " ", line).split()
            vect = np.array([0, float(genders[wordList[0]]), ranks[wordList[1]], float(wordList[2]), degrees[wordList[3]], float(wordList[4])]) #gender, rank, yr, degree received, yd
            A = np.append(A, [vect], axis=0)
            y = np.append(y, [float(wordList[5])], axis=0)
            length += 1
    return np.delete(A, 0, axis = 0), np.delete(y, 0), length

#feed forward into the network
#sigmoid activation network
def feed_forward_sigmoid(A, syn0, width):
    l0 = A
    l1 = np.array([sigmoid(np.dot(l0, syn0[:,1]))]) 
    for iter_width in range(width-1):
        l1 = np.append(l1, [sigmoid(np.dot(l0, syn0[:,iter_width]))],axis = 0) # Sigmoid activation function nodes
    return l1

#linear activation network
def feed_forward_linear(A, syn0, width):
    l0 = A
    l1 = np.array([sigmoid(np.dot(l0, syn0[:,1]))]) 
    for iter_width in range(width-1):
        # if(func[iter_width] < 0.5):
        l1 = np.append(l1, [linear(np.dot(l0, syn0[:,iter_width]))],axis = 0) # Linear activation function nodes
        # else:
        # l1 = np.append(l1, [sigmoid(np.dot(l0, syn0[:,iter_width]))],axis = 0) # Sigmoid activation function nodes
    return l1

#softplus activation network
def feed_forward_softplus(A, syn0, width):
    l0 = A
    l1 = np.array([softplus(np.dot(l0, syn0[:,1]))]) 
    for iter_width in range(width-1):
        # if(func[iter_width] < 0.5):
        l1 = np.append(l1, [softplus(np.dot(l0, syn0[:,iter_width+1]))],axis = 0) # Linear activation function nodes
        # else:
        # l1 = np.append(l1, [sigmoid(np.dot(l0, syn0[:,iter_width]))],axis = 0) # Sigmoid activation function nodes
    return l1

#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#linear function
def linear(x):
    return x

#softplus function
def softplus(x):
    return np.log(1+np.exp(x))

#read the training dataset
[A, y, length] = inp('salary.dat')
#randomized input weights
syn0 = np.random.normal(size = (6, width))
out = feed_forward_sigmoid(A, syn0, width)
w = np.linalg.lstsq(out.T, y)[0] #least square learning on the output weight of random layer

#read the test dataset
[A, y, length] = inp("salary_test.dat")
#feed test data into network
out = feed_forward_sigmoid(A, syn0, width)
print ('least square weight')
print w
print ('feed forward result')
print np.dot(out.T, w)
#calculate error
print(np.abs(np.average(np.dot(out.T, w) - y)))