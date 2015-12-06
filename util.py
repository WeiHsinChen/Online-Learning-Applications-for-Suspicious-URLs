import scipy.io as sio


def load_url(path='url.mat'):
    url = sio.loadmat(path)
    n_url = {i: {'data': url['Day%d' % i]['data'][0][0],
             'labels': url['Day%d' % i]['labels'][0][0]}
             for i in range(120)}
    return n_url
