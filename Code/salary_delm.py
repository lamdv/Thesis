import numpy as np
import re

#dictionary
genders = {'male':1, 'female':-1}
ranks = {"full":1, "associate":0, "assistant":-1}
degrees = {"masters":1, "doctorate":-1}

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
def feed_forward(A, syn, activ = "sigmoid"):
    if activ == "sigmoid":
        l1 = sigmoid(np.dot(A, syn))
        return l1
    elif activ == "linear":
        l1 = np.dot(A, syn0)
        return l1
    else:
        l1 = softplus(np.dot(A, syn)) # Sigmoid activation function nodes
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

def elm(A, y, seed = None, activ = "sigmoid", width = 1000):
    np.random.seed(seed)
    #func = np.random.rand(width, 1) #randomly assign activation function to the nodes of the network

    #read the training dataset
    #randomized input weights
    syn0 = np.random.normal(loc = 0.0, scale = 0.5,size = (7, width))
    h = feed_forward(A, syn0, activ)
    w = np.linalg.lstsq(h, y)[0] #least square learning on the output weight of random layer

    # #print('',err)
    return [syn0, w]


[A, y, length] = inp('salary.dat')
y = np.array([y]).T
A = np.append(A, y, axis = 1)
print(np.shape(A))

#training
[syn0, w0] = elm(A, A, width = 400)
A = np.dot(feed_forward(A, syn0), w0)

# h01 = np.dot(feed_forward(A, syn0), w0)
# [syn1, w1] = elm(h01, h01, width = 100)
# h12 = np.dot(feed_forward(A, syn1), w1)
# [syn2, w2] = elm(h12, h12, width = 100)
# h23 = np.dot(feed_forward(A, syn2), w2)
# [syn3, w3] = elm(h23, h23, width = 100)

#read the test dataset
[A, y, length] = inp("salary_test.dat")
line = np.zeros([length,1])
A = np.append(A, line, axis = 1)
_y = np.dot(feed_forward(A,syn0), w0)[:,6]

# #feed test data into network
# _y = np.dot(feed_forward(np.dot(feed_forward(np.dot(feed_forward(np.dot(feed_forward(A, syn0), w0), syn1), w1), syn2), w2), syn3), w3)

#calculate error
print(np.average(abs(y-_y)))