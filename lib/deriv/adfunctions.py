import numpy as np
from lib.deriv.adtools import cstest

def squeeze(A,axis=None):
    A = np.squeeze(A,axis=axis)
    return A.item() if A.ndim==0 else A

def logsumexp(X, axis=0, keepdims = False, deriv=False):
    """
    This is a complex-step friendly version of logsumexp. 
    """
    maxX = np.real(X).max(axis=axis,keepdims=True)
    Y = np.log(np.exp(X - maxX).sum(axis=axis,keepdims=True))
    Y += maxX
    Yshape = Y.shape
    if not deriv:
        return Y if keepdims else squeeze(Y,axis=axis) 

    S = np.exp(X - Y)  # softmax
    def back(dY=1):
        if np.isscalar(dY) :
            if dY == 1: return S
            return dY*S
        return  dY.reshape(Yshape) * S


    return Y if keepdims else squeeze(Y,axis=axis), back
            
     
        
if __name__ == "__main__":
    print("Running test script for module adfunctions\n")
    
    from numpy.random import randn
    
    
    print("Testing logsumexp")
    X = randn(2,3)
    delta = cstest(logsumexp,X,keepdims=False,axis=0)
    print(delta)

    delta = cstest(logsumexp,X,keepdims=False,axis=1)
    print(delta)

    delta = cstest(logsumexp,X,keepdims=True,axis=0)
    print(delta)


    delta = cstest(logsumexp,X,keepdims=True,axis=1)
    print(delta)
    
    X = randn(3)
    delta = cstest(logsumexp,X,keepdims=False)
    print(delta)
    
    delta = cstest(logsumexp,X,keepdims=True)
    print(delta)
    