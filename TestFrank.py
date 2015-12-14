

import numpy as np
# from StochasticGradient import SG
from scipy.sparse import csc_matrix
from util import load_url

# Load URLs
# sg = SG()

# day1X = csc_matrix([[1,0,1,1,0], [0,0,1,0,0]])
# day1Y = csc_matrix([1, 0])
# day1Y = day1Y.T

# print day1X[0, :]
# print day1Y[:,1]
# data = {0: {'data':day1X, 'labels':day1Y},1: {'data':day1X, 'labels':day1Y}, 2: {'data':day1X, 'labels':day1Y}}


# (cumuErrorLs, cumuFPLs) = sg.calculate(0.01, 100, data)
# np.savetxt('cumuErrorLs.txt', cumuErrorLs)    
# np.savetxt('cumuFPLs.txt', cumuFPLs)
# print (cumuErrorLs, cumuFPLs)


# from libsvm.python.svmutil import *

# Set parameter
# use SVM with Linear kernel and C = 100
# param = svm_parameter('-t 0 -c 100 -b 1')

# SVM-once, train with only data on day 1
# y, x = data[0]['labels'], data[0]['data']
# prob  = svm_problem(y, x)
# m = svm_train(prob, param)


from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from StochasticGradientWithPCA import StochasticGradientWithPCA
from NeuronNetwork import NeuronNetwork
from scipy import sparse
from Perceptron import Perceptron
import datetime


def testPCA():
    data = load_url()
    SG = StochasticGradientWithPCA()
    X  = data[0]['data']
    (U, S, V) = svds(X.T, 100)
    np.savetxt('U.txt', U)
    np.savetxt('S.txt', S)
    np.savetxt('V.txt', V)
    U = csc_matrix(U)
    del S
    del V
    cumuErrorLs = []
    cumuFPLs = []
    print ('start loop')
    for d in range(0, 100):
        print ('day: ', d)
        X = data[d]['data']
        (false, false_negative, w) = SG.run(X*U, data[d]['labels'])
        cumuErrorLs.append(false)
        cumuFPLs.append(false_negative)
        (U, S, V) = svds(X.T, 100)
        U = csc_matrix(U)
        del S
        del V

    np.savetxt('cumuErrorPCA.txt', cumuErrorLs) 
    np.savetxt('cumuFPPCA.txt', cumuFPLs)
    np.savetxt('wPCA.txt', w)


# testPCA()

# a = datetime.datetime.now()
# def testPCA():
#   sgPca = SGWithPCA()
#   (cumuErrorLsPCA, cumuFPLsPCA, wPCA) = sgPca.calculate(0.01, 100, data, csc_matrix(U))
#   b = datetime.datetime.now()
#   print 'time cost: ', (b-a)
#   np.savetxt('cumuErrorLsPCA.txt', cumuErrorLsPCA)    
#   np.savetxt('cumuFPLsPCA.txt', cumuFPLsPCA)
#   np.savetxt('wPCA.txt', wPCA)
#   np.savetxt('PCATime.txt', (b-a))

def testNN():
    data = load_url() 
    cum_total = 0
    cum_error = 0
    cum_fn = 0
    cum_ErrorLs = []
    cum_FNLs = []
    NN = NeuronNetwork()
    for d in range(120):
        X, y = data[d]['data'], data[d]['labels']
        [nEx, nF] = X.shape
        (error, fn) = NN.run(X, y)

        cum_total += nEx
        cum_error += error
        cum_fn += fn
        cum_ErrorLs.append(cum_error/(cum_total+1))
        cum_FNLs.append(cum_fn/(cum_total+1))
        np.savetxt('cumuErrorNN.txt', cum_ErrorLs)
        np.savetxt('cumuFnNN.txt', cum_FNLs)

# testNN()

def fixVariablePerceptron():
    # for 
    data = load_url() 
    tmp = sparse.find(csc_matrix(np.absolute(data[0]['data']).sum(axis=0)))
    pc = Perceptron()

    cum_total = 0
    cum_error = 0
    cum_fn = 0
    cum_ErrorLs = []
    cum_FNLs = []
    for d in range(120):
        X, y = data[d]['data'], data[d]['labels']
        [nEx, nF] = X.shape
        (error, fn) = pc.run(X[:, tmp[1]], y)

        cum_total += nEx
        cum_error += error
        cum_fn += fn
        cum_ErrorLs.append(cum_error/(cum_total+1))
        cum_FNLs.append(cum_fn/(cum_total+1))
        np.savetxt('cumuErrorPerceptronFixed.txt', cum_ErrorLs)
        np.savetxt('cumuFNPerceptronFixed.txt', cum_FNLs)
        

    # print (tmp[0])
    # print ('X[:, 3231954]', X[:, 3231954].sum(axis=0))
    # print ('X[:, 3231953]', X[:, 3231953].sum(axis=0))

# fixVariablePerceptron()
from PassiveAggressive import PassiveAggressive
def testPA():
    # for 
    data = load_url() 
    pa = PassiveAggressive()

    cum_total = 0
    cum_error = 0
    cum_fn = 0
    cum_ErrorLs = []
    cum_FNLs = []
    alphaLs = []
    for d in range(120):
        X, y = data[d]['data'], data[d]['labels']
        [nEx, nF] = X.shape
        (error, fn, alphas) = pa.run(X, y)

        cum_total += nEx
        cum_error += error
        cum_fn += fn
        cum_ErrorLs.append(cum_error/(cum_total+1))
        cum_FNLs.append(cum_fn/(cum_total+1))
        alphaLs.append('day '+str(d))
        alphaLs.extend(alphas)
        np.savetxt('cumuErrorPA.txt', cum_ErrorLs)
        np.savetxt('cumuFNPA.txt', cum_FNLs)
        np.savetxt('alphaLs.txt', alphaLs)

testPA()
    