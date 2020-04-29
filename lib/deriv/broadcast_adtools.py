import numpy as np

def sum2sh(x,to_shape):
    x_shape = np.shape(x)
    if x_shape == to_shape: return x
    n = len(to_shape)
    if n == 0: return np.sum(x)
    m = len(x_shape)
    assert m >= n
    i_shape = (1,)*(m-n) + to_shape  # if m==n: i_shape = to_shape
    axes = tuple(i for x,d,i in zip(x_shape,i_shape,range(m)) if x > d and d==1)
    x = np.sum(x,axis=axes,keepdims = True)
    if m>n: x = x.reshape(to_shape)
    assert x.shape == to_shape
    return x

def sum2shape(x,to_shape):
    x_shape = np.shape(x)
    if x_shape == to_shape: return x
    n = len(to_shape)
    if n == 0: return np.sum(x)
    m = len(x_shape)
    axes = tuple(i for x,d,i in zip(x_shape,to_shape,range(m)) if x > d and d==1)
    x = np.sum(x,axis=axes,keepdims = True)
    assert x.shape == to_shape
    return x

    
    
    



    