from numpy import zeros
# x is k features by N instances
# y is 1 label in {0,1} by N instances
# w: weight vector 'w' after learning


class Perceptron:
    # w = csc_matrix(np.random.rand(k, 1))
    def __init__(self, w=None):
        self.w = w
    def reset(self):
        self.w = None
    def run(self, x, y):
        # return count of false and false negative
        cumu_false = 0.0
        cumu_false_negative = 0.0
        N, k = x.shape
        if self.w is None:
            self.w = zeros((k, 1))
        for i in xrange(N):
            xi = x[i, :].T
            yi = y[i][0]
            M = yi * xi.T.dot(self.w)
            mistake = 1 if (M.sum() <= 0) else 0
            false_negative = 1 if yi == 1 and mistake else 0
            cumu_false += mistake
            cumu_false_negative += false_negative
            if M.sum() <= 0:
                self.w = self.w + yi * xi
        return (cumu_false, cumu_false_negative)


