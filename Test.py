

import numpy as np
from StochasticGradient import SG
from scipy.sparse import csc_matrix
from util import load_url
data = load_url()

sg = SG()

day1X = csc_matrix([[1,0,1,1,0], [0,0,1,0,0]])
day1Y = csc_matrix([1, 0])
day1Y = day1Y.T


# print day1X[0, :]
# print day1Y[:,1]
# data = {0: {'data':day1X, 'labels':day1Y},1: {'data':day1X, 'labels':day1Y}, 2: {'data':day1X, 'labels':day1Y}}
(cumuErrorLs, cumuFPLs) = sg.calculate(0.01, 100, data)

print (cumuErrorLs, cumuFPLs)