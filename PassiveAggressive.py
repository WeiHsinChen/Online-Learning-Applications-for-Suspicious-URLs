from numpy import zeros
from math import sqrt


class PassiveAggressive():
    def __init__(self, w=None):
        self.w = w

    def run(self, x, y):
        N, k = x.shape
        if self.w is None:
            self.w = zeros((k, 1))
        cumu_false = 0.0
        cumu_false_negative = 0.0
        for i in xrange(N):
            xi = x[i, :].T
            yi = y[i][0]
            M = yi * (xi.T * self.w)
            mistake = 1 if (M <= 0) else 0
            false_negative = 1 if yi == 1 and mistake else 0
            cumu_false += mistake
            cumu_false_negative += false_negative
            if M < 1:
                x_norm = sqrt((x.T * x).sum())
                if x_norm != 0:
                    alpha = (1 - M) / (x_norm ** 2)
                    self.w += xi * alpha * yi
        return (cumu_false, cumu_false_negative)
