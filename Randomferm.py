import random
FERN_NUM = 10
S_DEPTH = 10
CLASS_NUM = 2
NUMBER_OF_FEATURE = FERN_NUM * S_DEPTH
RANDOM_SEQ = range(NUMBER_OF_FEATURE)
random.shuffle(RANDOM_SEQ)
FEATURE_SET = [RANDOM_SEQ[i * S_DEPTH:(i + 1) * S_DEPTH]
               for i in range(FERN_NUM)]
REGULARIZE_TERM = 1


def sample_with_replace(population):
    while True:
        yield random.choice(population)


class Randomferm(object):
    """docstring foRandom_fermme"""
    def __init__(self, class_num, depth, fern_num, number_of_feature):
        self.class_num = class_num
        self.fern_num = fern_num
        self.depth = depth
        self.ferms = []
        self.number_of_feature = number_of_feature
        self.possiblity_of_class = [0 for _ in xrange(class_num)]

    def _init_ferms(self):
        idx_set = range(self.number_of_feature)
        for _ in xrange(self.fern_num):
            feature_idx_list = []
            while len(feature_idx_list) != self.depth:
                f_idx = sample_with_replace(idx_set).next()
                if f_idx not in feature_idx_list:
                    feature_idx_list.append(f_idx)
            self.ferms.append(
                Ferm(feature_idx_list,
                     self.class_num,
                     self.depth))
        return

    def train(self, xi, yi):
        self.possiblity_of_class[yi] += 1
        if not self.ferms:
            self._init_ferms()
        for ferm in self.ferms:
            ferm.count(xi, yi)
        return

    def predict(self, xi):
        p = self.possiblity_of_class[:]
        for i in range(len(p)):
            if p[i] == 0:
                p[i] = -10e100
            else:
                p[i] = math.log(p[i])
        for ferm in self.ferms:
            for idx, ferm_p in enumerate(ferm.get_p(xi)):
                p[idx] += math.log(ferm_p)
        return p.index(max(p))


class Ferm(object):
    def __init__(self, feature_idx_list, class_num, depth):
        self.class_num = class_num
        self.feature_idx_list = feature_idx_list
        self.p = [[REGULARIZE_TERM for _ in xrange(2 ** depth)]
                  for _ in range(class_num)]
        self.normalize_sum = 1.0 * REGULARIZE_TERM * class_num * 2 ** depth

    def _calc_binary_code(self, xi):
        binary_code = 0
        for idx, fi in enumerate(self.feature_idx_list):
            binary_code = binary_code + xi[fi].sum() * 2 ** idx
        return int(binary_code)

    def count(self, xi, class_i):
        binary_code = self._calc_binary_code(xi)
        self.p[class_i][binary_code] += 1
        self.normalize_sum += 1
        return

    def get_p(self, xi):
        binary_code = self._calc_binary_code(xi)
        p = [self.p[class_i][binary_code] / self.normalize_sum
             for class_i in range(self.class_num)]
        return p


def random_ferm(X, Y):
    N, k = X.shape
    np.place(Y, Y == -1, 0)
    err = zeros((N, 1))
    rf = Randomferm(2, 10, 100000, k)
    for i in xrange(N):
        x = X[i, :].T
        y = Y[i][0]
        last_err = err[i - 1] if i > 0 else 0
        mistake = 1 if rf.predict(x) != y else 0
        err[i] = last_err + mistake
        print "%d, %d" % (i, err[i])
        rf.train(x, y)
    return err