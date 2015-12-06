from util import load_url
data = load_url()

import numpy as np
from cw import cw

cw_xtest = cw()

day1X = np.matrix([[1,0,1,1,0], [0,0,1,0,0]])
day1Y = np.matrix([1, 0])



# print day1X[0, :]
# print day1Y[:,1]
#data = {0: {'data':day1X, 'labels':day1Y},1: {'data':day1X, 'labels':day1Y}, 2: {'data':day1X, 'labels':day1Y}}
eta = 0.95
sigma_format = 'diag'
(cumuErrorLs, cumuFPLs) = cw_xtest.calculate( 3, data, eta, sigma_format)

print (cumuErrorLs, cumuFPLs)
