import numpy as np


def softplus(x):
    """log( 1 + exp(x) )"""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def softplusinv(y):
    """x = log( exp(y) -1 )"""
    #return np.log(np.expm1(y))
    s = np.sign(y)
    with np.errstate(divide='ignore'):
        return np.log(-s*np.expm1(-np.abs(y))) + np.maximum(y,0)

