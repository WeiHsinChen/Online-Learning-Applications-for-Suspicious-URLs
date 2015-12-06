from util import load_url
import scipy.sparse as ss
from numpy import ones, zeros, multiply
import time
# X is k features by N instances
# Y is 1 label in {0,1} by N instances
# params is struct containing options

# err: cumulative mistakes after each example
# mu: weight vector 'mu' after learning
# sigma: struct containing covariance/precision (inv. covariance) matrix after learning
# mem: memory consumption


def perceptron(X, Y, params={}):
    start = time.clock()
    N, k = X.shape
    mu = (params['mu'] if 'mu' in params.keys()
          else ss.csr_matrix(zeros((k, 1))))
    # mu = params['mu'] if 'mu' in params.keys() else zeros((k, 1))

    # average = (params['average'] if 'average' in params.keys()
    #            else False)
    err = zeros((N, 1))
    # sigma = ss.diags(ones(k), 0, format='csc')
    for i in xrange(N):
        x = X[i, :].T
        y = Y[i][0]
        M = y * (x.T * mu)
        last_err = err[i - 1] if i > 0 else 0
        mistake = 1 if (M <= 0) else 0
        err[i] = last_err + mistake
        if M <= 0:
            mu = mu + y * x
    end = time.clock()
    print "%.2fs" % (end - start)