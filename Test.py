

import numpy as np
from StochasticGradient import SG

sg = SG()

day1X = np.matrix([[1,0,1,1,0], [0,0,1,0,0]])
day1Y = np.matrix([1, 0])



# print day1X[0, :]
# print day1Y[:,1]
data = {0: {'data':day1X, 'labels':day1Y},1: {'data':day1X, 'labels':day1Y}, 2: {'data':day1X, 'labels':day1Y}}
(cumuErrorLs, cumuFPLs) = sg.calculate(7, 3, data)

print (cumuErrorLs, cumuFPLs)