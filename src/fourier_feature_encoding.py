import numpy as np


def multiscale(x, scales):
    return np.hstack([x.reshape(-1, 1)/pow(2., i) for i in scales])


def encode_scalar_column(x, scales=[-1, 0, 1, 2, 3, 4, 5, 6]):
    return np.hstack([np.sin(multiscale(x, scales)), np.cos(multiscale(x, scales))])
