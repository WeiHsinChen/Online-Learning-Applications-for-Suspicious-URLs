import sys
import numpy as np
import math
from scipy import sparse
from scipy.special import erf, erfinv


class cw:
    def calculate(self, days, data, eta, sigma_format):
        if days == 0:
            return None
        # init parameter
        # prepare probability eta(default is 0.95) and phi
        phi = math.sqrt(2) * erfinv(2 * eta - 1)
        psi = 1 + phi ** 2 / 2.0
        xi = 1 + phi ** 2

        # 0. Use day 1 to extract dimention needed
        (nEx, nF) = np.shape(data[1]['data'])

        # 1. initialize mu, sigma
        # non sparse version
        # mu    = np.matrix(np.zeros((nF,1)))
        # sigma = np.matrix(np.ones((nF,1)))
        # sparse version
        mu = np.zeros((nF, 1))
        sigma = np.eye(nF)

        # calculate error rate
        cumuErrorLs = []
        cumuFNLs = []
        cumuTotal = 0.0
        cumuFalse = 0.0
        cumuFalseNegative = 0.0
        for d in xrange(days):
            X, Y = data[d]['data'], data[d]['labels']
            print 'day:', d
            # Start CW
            (nEx, nF) = np.shape(X)
            for i in xrange(nEx):
                if i % 100 == 0:
                    print 'step: ', i
                    print 'Cumulative Error Rate', cumuFalse / (cumuTotal + 1)
                    print 'Cumulative False Negative Rate', cumuFalseNegative / (cumuTotal + 1)
                x = X[i, :].T
                y = Y[i, :][0]

                #  Calculate Error
                predict = 1 if np.sign((mu.T * x).sum()) >= 0 else -1
                if predict != y:
                    cumuFalse += 1
                    if y == 1:
                        cumuFalseNegative += 1
                cumuTotal += 1
                # Run CW
                # M = y*(x.T*mu)
                M = y * ((mu.T * x).sum())
                # calculate sigma, sigma_x
                # 1. sigma_x
                sigma_x = sigma.dot(x)
                # 3. V
                # V = x.T*sigma_x
                V = ((x.T).dot(sigma_x)).sum()
                # case from 'stdef'
                if M < phi * math.sqrt(V):
                    alpha = (
                        -M * psi
                        + math.sqrt(
                            (M ** 2) * (phi ** 4) / 4.0
                            + V * (phi ** 2) * xi)
                    ) / (V * xi)
                    # if isreal(alpha) & ~isnan(alpha) & ~isinfinite(alpha)
                    if np.isfinite(alpha):
                        sqrtU = (
                            -alpha * V * phi
                            + math.sqrt(
                                (alpha ** 2) * (V ** 2) * (phi ** 2)
                                + 4 * V)
                        ) / 2
                        # for full sigma representation
                        beta = (alpha * phi) / (sqrtU + V * alpha * phi)
                        # mu = mu + np.multiply(alpha*y,sigma_x)
                        mu = mu + sigma_x.multiply(alpha * y)
                        # if average
                        #   v = v - i*alpha*y*sigma_x;
                        # end
                        # sigma = sigma - beta*(sigma_x.^2)
                        # sigma = sigma - np.multiply(beta,np.power(sigma_x,2))
                        sigma_x_cov = sigma_x * sigma_x.T
                        sigma = sigma - sigma_x_cov.multiply(beta)
            cumuErrorLs.append(cumuFalse/cumuTotal)
            cumuFNLs.append(cumuFalseNegative/cumuTotal)
        return (cumuErrorLs, cumuFNLs)
