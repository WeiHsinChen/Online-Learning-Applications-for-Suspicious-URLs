import scipy.io as sio
import numpy as np


def load_url(path='url.mat'):
    url = sio.loadmat(path)
    n_url = {i: {'data': url['Day%d' % i]['data'][0][0],
             'labels': url['Day%d' % i]['labels'][0][0].astype(np.int8)}
             for i in range(120)}
    for i in range(120):
        np.place(n_url[i]['labels'], n_url[i]['labels'] == 0, -1)
    return n_url
