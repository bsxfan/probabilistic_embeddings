import numpy as np

import lib.combin.partitions as part
from lib.deriv.adfunctions import logsumexp


class NTupleModel:
    
    def __init__(self,n,prior,likelihood):
        """
        Constructs an object to model all Bn ways to cluster an n-tuple of inputs, 
        by using a given clustering prior (e.g. CRP) and a likelihood model. 
        """
        self.n = n
        P, S = part.partitions_and_subsets(n)
        self.P, self.S = P, S
        self.priortable = prior.logprobtable(P,S)
        self.likelihood = likelihood
        self.index = part.create_index(n) # maps restricted growth string
        
    def  logposterior(self,X,B,rgs=None):
        """
        Returns log P(rgs | X,B), where rgs is a stricted growth string of 
        length n, so that rgs identifies one of the partitions of the given
        set of n inputs. The set of n inputs is specified as the set of 
        probabilistic embeddings given by X,B. 
        
        If rgs is omitted, the whole (unnormalized) log-posterior distribution
        is returned as a vector of length Bn. The order of the elements 
        corresponds to the iterator: lib.combin.PartitionsOf(n). 
        
        In summary:
            - If rgs is specified one element of the normalized log-posterior is 
              returned. 
            - If rgs is not specified, the whole distribution is returned
               in unnormalized log form. The missing normalizer can be found 
               using logsumexp.
               
        The parameters X,B are both of shape (d,n), where d must agree
        with the likelihood model that was passed to the constructor of this 
        object. And n must agree with the value of n that was passed to the 
        constructor.
                
        X: (d,n) set of n, d-dimensional embedding point estimates
        B: (d,n) set of n, d-dimensional non-negative embedding estimate 
                 precisions.
                 
        rgs: restricted growth string (set of integer labels) of length n, to
             specify which element of the posterior distribution to return.          
        
        """
        assert X.shape == B.shape
        n,d = X.shape
        assert n == self.n
        subset_LLH = self.likelihood.logLH(self.S,X,B)
        logpost = self.P @ subset_LLH + self.priortable
        if rgs is None: return logpost
        
        i = self.index(rgs)
        return logpost[i] - logsumexp(logpost)
    
if __name__ == "__main__":
    print("Running test script for module ntuplemodel\n")
    
    from lib.prob.crp import CRP
    alpha, beta = 3, 0.7
    crp = CRP(alpha,beta)

    from numpy.random import randn
    from diag2covplda import model as PLDA
    dim = 10    
    w = np.exp(randn(dim))
    plda = PLDA(w)
    
    n = 3
    tmodel = NTupleModel(n,crp,plda)
    
    X = randn(n,dim)
    B = randn(n,dim)**2
    L = [np.exp(tmodel.logposterior(X,B,rgs)) for rgs in PartitionsOf(n)]
    print(L)
    print(np.array(L).sum())
    
    
    