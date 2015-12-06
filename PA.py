from util import load_url
from numpy import zeros
from math import sqrt
# X is k features by N instances
# Y is 1 label in {0,1} by N instances
# params is struct containing options

# err: cumulative mistakes after each example
# mu: weight vector 'mu' after learning
# sigma: struct containing covariance/precision (inv. covariance) matrix after learning
# mem: memory consumption


def PassiveAggressive(X, Y, params={}):
    N, k = X.shape
    mu = params['mu'] if 'mu' in params.keys() else zeros((k, 1))

    err = zeros((N, 1))
    # sigma = ss.diags(ones(k), 0, format='csc')
    for i in xrange(N):
        x = X[i, :].T
        y = Y[i][0]
        M = y * (x.T * mu)
        last_err = err[i - 1] if i > 0 else 0
        mistake = 1 if (M <= 0) else 0
        err[i] = last_err + mistake
        if M < 1:
            x_norm = sqrt((x.T * x).sum())
            if x_norm != 0:
                alpha = (1 - M) / (x_norm ** 2)
                mu = mu + x * alpha * y
