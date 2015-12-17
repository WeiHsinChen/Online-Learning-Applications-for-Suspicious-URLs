from Perceptron import Perceptron
from StochasticGradient import StochasticGradient
from PassiveAggressive import PassiveAggressive
from ConfidenceWeighted import ConfidenceWeighted
from util import load_url
from sklearn.svm import LinearSVC
from scipy.sparse import vstack
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from time import time
import numpy as np


def run_online_model_experiment():
    data = load_url()
    model_list = [Perceptron(), StochasticGradient(),
                  PassiveAggressive(), ConfidenceWeighted()]
    model_name = ['Perceptron', 'StochasticGradient',
                  'PassiveAggressive', 'ConfidenceWeighted']
    model_name = ['PassiveAggressive']
    model_list = [PassiveAggressive()]
    for idx, model in enumerate(model_list):
        cumu_false_total = 0.0
        cumu_false_negative_total = 0.0
        total_count = 0
        file_false = open('%s_result_f' % model_name[idx], 'w')
        file_false_negative = open('%s_result_fn' % model_name[idx], 'w')
        for i in xrange(len(data)):
            print '%s day(%d)' % (model_name[idx], i)
            x = data[i]['data']
            y = data[i]['labels']
            cumu_false, cumu_false_negative = model.run(x, y)
            cumu_false_total += cumu_false
            cumu_false_negative_total += cumu_false_negative
            total_count += len(y)
            file_false.write("%f\n" % cumu_false_total / total_count)
            file_false_negative.write("%f\n" %
                                      cumu_false_negative_total / total_count)
        file_false.close()
        file_false_negative.close()


def run_speed_test():
    data = load_url()
    x = data[0]['data'][:1000]
    y = data[0]['labels'][:1000]
    model_list = [Perceptron(), StochasticGradient(),
                  PassiveAggressive(), ConfidenceWeighted()]
    model_name = ['Perceptron', 'StochasticGradient',
                  'PassiveAggressive', 'ConfidenceWeighted']
    for idx, model in enumerate(model_list):
        start = time()
        model.run(x[:1000], y[:1000])
        print "%s run %f sec" % (model_name[idx], time() - start)
# Perceptron run 26.424507 sec
# StochasticGradient run 23.953401 sec
# PassiveAggressive run 764.576739 sec
# ConfidenceWeighted run 2628.595875 sec


def run_pca_speed_test():
    data = load_url()
    (U, S, V) = svds(data[0]['data'].T, 100)
    U = csc_matrix(U)
    x = data[0]['data'][:1000]
    y = data[0]['labels'][:1000]
    model_list = [Perceptron(), StochasticGradient(),
                  PassiveAggressive(), ConfidenceWeighted()]
    model_name = ['Perceptron', 'StochasticGradient',
                  'PassiveAggressive', 'ConfidenceWeighted']
    pca_x = x.dot(U)
    for idx, model in enumerate(model_list):
        start = time()
        model.run(x, y)
        print "%s run %f sec (no_pca)" % (model_name[idx], time() - start)
        start = time()
        for i in range(1, x.shape[0]):
            new_xi = x[i, :].dot(U)
        model.reset()
        model.run(pca_x, y)
        print "%s run %f sec (with_pca)" % (model_name[idx], time() - start)


def svm_once(data):
    cumu_false_total = 0.0
    cumu_false_negative_total = 0.0
    total_count = 0
    x = data[0]['data']
    y = np.ravel(data[0]['labels'])
    svc = LinearSVC(C=100)
    svc.fit(x, y.tolist())
    file_false = open('svm_once_f.txt', 'w')
    file_false_negative = open('svm_once_fn.txt', 'w')
    for i in xrange(1, 100):
        print 'day(%d)' % i
        x = data[i]['data']
        y = np.ravel(data[i]['labels'])
        p_y = svc.predict(x)
        cumu_false = 0
        cumu_false_negative = 0
        for idx, f in enumerate(y == p_y):
            if not f:
                cumu_false += 1
                if y[idx] == 1:
                    cumu_false_negative += 1
        cumu_false_total += cumu_false
        cumu_false_negative_total += cumu_false_negative
        total_count += len(y)
        file_false.write("%f\n" % (cumu_false_total / total_count))
        file_false_negative.write("%f\n" %
                                  (cumu_false_negative_total / total_count))
    file_false.close()
    file_false_negative.close()


def svm_daily(data):
    cumu_false_total = 0.0
    cumu_false_negative_total = 0.0
    total_count = 0
    x = data[0]['data']
    y = np.ravel(data[0]['labels'])
    svc = LinearSVC(C=100)
    svc.fit(x, y.tolist())
    file_false = open('svm_daily_f.txt', 'w')
    file_false_negative = open('svm_daily_fn.txt', 'w')
    for i in xrange(1, 100):
        print 'day(%d)' % i
        x = data[i]['data']
        y = np.ravel(data[i]['labels'])
        p_y = svc.predict(x)
        cumu_false = 0
        cumu_false_negative = 0
        for idx, f in enumerate(y == p_y):
            if not f:
                cumu_false += 1
                if y[idx] == 1:
                    cumu_false_negative += 1
        cumu_false_total += cumu_false
        cumu_false_negative_total += cumu_false_negative
        total_count += len(y)
        if i > 40 or i < 35:
            svc.fit(x, y.tolist())
        file_false.write("%f\n" % (cumu_false_total / total_count))
        file_false_negative.write("%f\n" %
                                  (cumu_false_negative_total / total_count))
    file_false.close()
    file_false_negative.close()


def svm_cum(data):
    cumu_false_total = 0.0
    cumu_false_negative_total = 0.0
    total_count = 0
    train_x = data[0]['data']
    train_y = np.ravel(data[0]['labels'])
    svc = LinearSVC(C=100)
    svc.fit(train_x, train_y.tolist())
    file_false = open('svm_cum_f.txt', 'w')
    file_false_negative = open('svm_cum_fn.txt', 'w')
    for i in xrange(1, 100):
        print 'day(%d)' % i
        x = data[i]['data']
        y = np.ravel(data[i]['labels'])
        p_y = svc.predict(x)
        cumu_false = 0
        cumu_false_negative = 0
        for idx, f in enumerate(y == p_y):
            if not f:
                cumu_false += 1
                if y[idx] == 1:
                    cumu_false_negative += 1
        cumu_false_total += cumu_false
        cumu_false_negative_total += cumu_false_negative
        total_count += len(y)
        train_x = vstack((train_x, x))
        train_y = np.concatenate((train_y, y))
        svc.fit(train_x, train_y.tolist())
        file_false.write("%f\n" % (cumu_false_total / total_count))
        file_false_negative.write("%f\n" %
                                  (cumu_false_negative_total / total_count))
    file_false.close()
    file_false_negative.close()


def run_all_svm_experiment():
    data = load_url()
    svm_once(data)
    svm_daily(data)
    svm_cum(data)

if __name__ == '__main__':
    run_online_model_experiment()
    run_speed_test()
    run_pca_speed_test()
    run_all_svm_experiment()
