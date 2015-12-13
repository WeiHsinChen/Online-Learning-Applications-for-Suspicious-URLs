import numpy as np
from scipy.stats import logistic
from scipy.sparse import csc_matrix
# x is k features by N instances
# y is 1 label in {0,1} by N instances
# w: weight vector 'w' after learning


class StochasticGradientWithPCA:
    # w = csc_matrix(np.random.rand(k, 1))
    def __init__(self, w=None, gamma=0.01, cumu_false=0.0, cumu_false_negative=0.0, cumu_data=1.0):
        self.w = w
        self.gamma = gamma
        self.cumu_false = cumu_false
        self.cumu_false_negative = cumu_false_negative
        self.cumu_data = cumu_data

    def run(self, x, y, U):
        # N: # of Examples, k: # of features
        (N, k) = x.shape
        (tmp, kNew) = U.shape
        UT = U.T

        if self.w is None:
            self.w = csc_matrix(np.zeros((kNew, 1)))
        # Start PA
        for i in range(N):
            if i % 100 == 0:
                print ('step: ', i)
                print ('Cumulative Error Rate', self.cumu_false / self.cumu_data)
                print ('Cumulative False Negative Rate', self.cumu_false_negative / self.cumu_data)
            xi = x[i, :].T
            yi = y[i, :][0]

            xiNew = UT.dot(xi)
            tmp = (self.w.T).dot(xiNew)
            prob_of_positive = logistic.cdf(tmp[0, 0])
            
            predict = 1 if prob_of_positive >= 0.5 else -1
            mistake = 1 if (predict != yi) else 0
            false_negative = 1 if yi == 1 and mistake else 0
            self.cumu_false += mistake
            self.cumu_false_negative += false_negative
            self.cumu_data += 1
            # update w
            self.w = self.w + self.gamma * xiNew * ((yi + 1) / 2 - prob_of_positive)

        return (self.cumu_false, self.cumu_false_negative, self.w)
