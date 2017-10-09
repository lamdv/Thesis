import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

PATH = "Code\\MNIST\\"
input_size = 28*28
width = 100

def input(f, mode):
    mdata = MNIST(f)
    if(mode == 'training'):
        return mdata.load_training()
    elif(mode == 'testing'):
        return mdata.load_testing()
    else:
        return null

def feed_forward(A, syn, activ = "sigmoid"):
    if activ == "sigmoid":
        l1 = sigmoid(np.dot(A, syn))
    elif activ == "linear":
        l1 = np.dot(A, syn0)
    else:
        l1 = softplus(np.dot(A, syn)) # Sigmoid activation function nodes
    return l1

def unfold(x):
    _y = np.matrix(x)
    y = np.zeros(10, _y.shape())
    for i in range(_y.shape()):
        y[_y[i], i] = 1
    return _y


#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#linear function
def linear(x):
    return x

#softplus function
def softplus(x):
    return np.log(1+np.exp(x))

def elm(A, y, seed = None, activ = "sigmoid", width = 500):
    np.random.seed(seed)
    #randomized input weights
    syn0 = np.random.normal(size = (input_size, width))
    h = feed_forward(A, syn0, activ)
    w = np.linalg.lstsq(h, y)[0] #least square learning on the output weight of random layer
    #print('',err)
    return syn0, w

def test(A, y, At, yt, width):
    #traing using training data
    syn0, w = elm(A, y, width = width)
    #feed test data into network
    h = feed_forward(At, syn0, "sigmoid")
    _y = np.dot(h, w)
    hit = np.abs(_y - yt)
    #calculate hit rate
    hit = np.where(hit < 0.5, 1, 0)
    return np.average(hit)

#read the trainimg dataset
A, y = input(PATH, "training")
y = unfold(y)
print(y)
#read the test dataset
At, yt = input(PATH, "testing")
yt = unfold(yt)
print("Finished reading dataset")

out = np.empty(7)
for i in range(7):
    width = (i+1)*100
    print("10 test of", width, "neurons wide network")
    result = np.empty(50)
    for j in range(50):
        result[j] = test(A, y, At, yt, width)
    out[i] = np.average(result)
    print(out[i])