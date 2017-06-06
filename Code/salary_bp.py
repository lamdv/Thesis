import numpy as np
import re

np.random.seed(1)

#network parameters
width = 10
alpha = 0.001

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
def feed_forward(A, syn0, width):
    l0 = A
    li = sigmoid(np.dot(l0, syn0))
    #for wided layer (width>1)
    #calculate the output of each node
    return li

def feed_backward(yp, y, width):
    lo = yp
    delta = lo*deriv_sigmoid(y)
    return delta

#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    return x*(1-x)

#softplus function
def softplus(x):
    return np.log(1+np.exp(x))

#read the training dataset
[A, y, length] = inp('salary.dat')
#randomized initial weights
syn0 = np.random.normal(size = (6,width))
syn1 = np.random.normal(size = (width, 1))

#train the network 10000 times
# for i in range(10000):
#feed forward
hid = feed_forward(A, syn0, width)
out = feed_forward(hid, syn1, 1)
#feed backward: error signal calculation
l2_err = out[0] - y
delta_1 = feed_backward(l2_err, out[0], 1)
print np.shape(delta_1)
delta_0 = feed_backward(np.dot(delta_1, syn0.T), syn0, width)
    # #weight update
    # syn1 += alpha*out.T.dot(delta_1)
    # syn0 += alpha*hid.T.dot(delta_0)