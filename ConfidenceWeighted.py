import numpy as np
import math
from scipy import sparse
from scipy.special import erfinv


class ConfidenceWeighted:
    def __init__(self, eta=0.95, w=None, sigma=None):
        self.eta = eta
        self.w = w
        self.sigma = None

    def run(self, x, y):
        phi = math.sqrt(2) * erfinv(2 * self.eta - 1)
        psi = 1 + phi ** 2 / 2.0
        xii = 1 + phi ** 2

        # 0. Use dayi 1 to exitract dimention needed
        (N, k) = x.shape

        # 1. initialize w, sigma
        # non sparse version
        # w    = np.matrixi(np.zeros((k,1)))
        # sigma = np.matrixi(np.ones((k,1)))
        # sparse version
        if self.w is None:
            self.w = np.zeros((k, 1))
        if self.sigma is None:
            self.sigma = sparse.eye(k)

        # calculate error rate
        cumu_false = 0.0
        cumu_false_negative = 0.0
        for i in xrange(N):
            if i % 100 == 0:
                print 'step: ', i
                print 'Cumulative Error Rate', cumu_false / (i + 1)
                print 'Cumulative False Negative Rate', cumu_false_negative / (i + 1)
            xi = x[i, :].T
            yi = y[i, :][0]

            #  Calculate Error
            if np.sign((self.w.T * xi).sum()) > 0:
                predict = 1
            else:
                predict = 1
            mistake = 1 if (predict != yi) else 0
            false_negative = 1 if yi == 1 and mistake else 0
            cumu_false += mistake
            cumu_false_negative += false_negative

            # Run CW
            # M = yi*(xi.T*mu)
            M = yi * ((self.w.T * xi).sum())
            # calculate sigma, sigma_xi
            sigma_xi = self.sigma.dot(xi)
            # V = xi.T*sigma_xi
            V = ((xi.T).dot(sigma_xi)).sum()
            # case from 'stdef'
            if M < phi * math.sqrt(V):
                alpha = (
                    -M * psi
                    + math.sqrt(
                        (M ** 2) * (phi ** 4) / 4.0
                        + V * (phi ** 2) * xii)
                ) / (V * xii)
                # if isreal(alpha) & ~isnan(alpha) & ~isinfinite(alpha)
                if np.isfinite(alpha):
                    sqrt_u = (
                        -alpha * V * phi
                        + math.sqrt(
                            (alpha ** 2) * (V ** 2) * (phi ** 2)
                            + 4 * V)
                    ) / 2
                    beta = (alpha * phi) / (sqrt_u + V * alpha * phi)
                    self.w = self.w + sigma_xi.multiply(alpha * yi)
                    sigma_xi_cov = sigma_xi * sigma_xi.T
                    self.sigma -= sigma_xi_cov.multiply(beta)

        return (cumu_false, cumu_false_negative)
