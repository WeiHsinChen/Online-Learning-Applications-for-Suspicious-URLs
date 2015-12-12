from Perceptron import Perceptron
from StochasticGradient import StochasticGradient
from PassiveAggressive import PassiveAggressive
from ConfidenceWeighted import ConfidenceWeighted
from util import load_url
from time import time


def test_speed():
    data = load_url()
    x = data[0]['data'][:100]
    y = data[0]['labels'][:100]
    model_list = [Perceptron(), StochasticGradient(),
                  PassiveAggressive(), ConfidenceWeighted()]
    model_name = ['Perceptron', 'StochasticGradient',
                  'PassiveAggressive', 'ConfidenceWeighted']
    for idx, model in enumerate(model_list):
        start = time()
        model.run(x[:100], y[:100])
        print "%s run %f sec" % (model_name[idx], time() - start)

test_speed()
