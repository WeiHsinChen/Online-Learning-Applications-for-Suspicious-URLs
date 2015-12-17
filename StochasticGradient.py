import numpy as np
from scipy.stats import logistic
from scipy.sparse import csc_matrix
# x is k features by N instances
# y is 1 label in {0,1} by N instances
# w: weight vector 'w' after learning


class StochasticGradient:
    # w = csc_matrix(np.random.rand(k, 1))
    def __init__(self, w=None, gamma=0.01):
        self.w = w
        self.gamma = gamma
    def reset(self):
        self.w = None
    def run(self, x, y):
        # N: # of Examples, k: # of features
        (N, k) = x.shape
        cumu_false = 0.0
        cumu_false_negative = 0.0
        if self.w is None:
            self.w = csc_matrix(np.zeros((k, 1)))
        # Start PA
        for i in range(N):
            if i % 100 == 0:
                print('step: ', i)
                print('Cumulative Error Rate in this day', cumu_false / (i + 1))
                print('Cumulative False Negative Rate in this day', cumu_false_negative / (i + 1))
            xi = x[i, :].T
            yi = y[i, :][0]
            tmp = (self.w.T).dot(xi)
            prob_of_positive = logistic.cdf(tmp[0, 0])
            predict = 1 if prob_of_positive >= 0.5 else -1
            mistake = 1 if (predict != yi) else 0
            false_negative = 1 if yi == 1 and mistake else 0
            cumu_false += mistake
            cumu_false_negative += false_negative
            # update w
            self.w = self.w + self.gamma * xi * ((yi + 1) / 2 - prob_of_positive)
        return (cumu_false, cumu_false_negative)
