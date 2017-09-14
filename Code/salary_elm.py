import numpy as np
import matplotlib.pyplot as plt
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
    elif activ == "linear":
        l1 = np.dot(A, syn0)
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

def elm(seed = None, activ = "sigmoid", width = 1000):
    np.random.seed(seed)
    #func = np.random.rand(width, 1) #randomly assign activation function to the nodes of the network

    #read the training dataset
    [A, y, length] = inp('salary.dat')
    #randomized input weights
    syn0 = np.random.normal(size = (6, width))
    h = feed_forward(A, syn0, activ)
    w = np.linalg.lstsq(h, y)[0] #least square learning on the output weight of random layer

    #read the test dataset
    [A, y, length] = inp("salary_test.dat")
    #feed test data into network
    h = feed_forward(A, syn0, activ)

    #calculate error
    err = np.abs(np.average(np.dot(h, w) - y))
    #print('',err)
    return err

# print ("100 randomized weight networks test")
# result = np.empty([int((4000-100)/100),3])
# for _width in range(100, 4000,100)
# :
#     _result = np.empty([100])
#     for i in range(100):
#         _result [i] = elm( activ = "softplus", width = _width)
#     print(_width, np.std(_result), np.average(_result))
#     np.append(result,[_width, np.std(_result), np.average(_result)])
# # plt.plot(result[:,0], result[:,1], "-", result[:,2], "o")
# plt.plot(result[:,0], result[:,1], '-')
# plt.show()
print (elm(activ = "softplus", width = 100))