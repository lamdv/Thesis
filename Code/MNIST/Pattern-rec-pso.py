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
    else:
        l1 = softplus(np.dot(A, syn)) # Softplus activation function nodes
    return l1


#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#softplus function
def softplus(x):
    return np.log(1+np.exp(x))

def elm(A, y, syn0, seed = None, activ = "sigmoid", width = 1000):
    np.random.seed(seed)
    h = feed_forward(A, syn0, activ)
    w = np.linalg.lstsq(h, y)[0]
    return w

def test(syn0, w):
    #feed test data into network
    h = feed_forward(At, syn0, "sigmoid")
    _y = np.dot(h, w_o)
    hit = np.abs(_y - yt)
    #calculate hit rate
    hit = np.where(hit < 0.5, 1, 0)
    return np.average(hit)

def find_direction(X, number_of_particle):
    best_particle = 0
    best_result = test(X[0])
    for i in range(1, number_of_particle):
        curr_result = test(X[i])
        if(curr_result<best_result):
            best_result = curr_result
            best_particle = i
    D = X - np.full([number_of_particle, 6, 250], X[best_particle])
    # D.fill(np.asscalar(X[best_particle]))
    return [D, best_particle, best_result]

def PSO(number_of_iter, number_of_particle):
    #spawn initial position and velocity
    X = np.random.uniform(-0.5, 0.5, size = [number_of_particle, 6, 250])
    V = np.random.uniform(-0.5, 0.5, size = [number_of_particle, 6, 250])
    #iteration:
    Dglobal = np.empty([number_of_particle,6,250])
    best_result = 1000000
    for iter in range(number_of_iter):
        [D, local_best_particle, local_best_result] = find_direction(X, number_of_particle)
        if(local_best_result < best_result):
            Dglobal = D
        V = V + D + Dglobal
        X = X + V
    best_particle = 0
    best_result = test(X[0])
    for i in range(1, number_of_particle):
        curr_result = test(X[i])
        if(curr_result<best_result):
            best_result = curr_result
            best_particle = i
    return X[best_particle]

#read the trainimg dataset
A, y = input(PATH, "training")
#read the test dataset
At, yt = input(PATH, "testing")
print("Finished reading dataset")
