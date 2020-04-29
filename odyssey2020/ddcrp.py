"""
Distance-dependent CRP:
http://www.jmlr.org/papers/volume12/blei11a/blei11a.pdf
"""

import numpy as np
from numpy.random import choice, rand 

class DDCRP:
    """
    Base class for distance dependent Chinese restaurant process. Provides
    sampling and log probability computation. An object of this class models
    partitions of a set of n elements, where n is fixed at construction.
    (This is unlike lib.prob.crp.CRP, which does not fix n.)
    """
    
    def __init__(self,P):
        """
        Constructs DDCRP, given an unnormalized (n,n) probability matrix. This 
        constructor normalizes the rows of P inplace.
        
        On entry, the off-diagonal elements of P must contain f(d_{ij}), 
        as defined in the paper. The diagonal elements must be alpha. 
        
        """
        m,n = P.shape
        assert m == n
        self.n = n
        P /= P.sum(axis=1,keepdims=True)
        self.P = P
        with np.errstate(divide='ignore'): 
            self.logP = np.log(P)
        self.ii = np.arange(n)
        
            
    def logprob(self,links):
        """
        Computes log P(links | this model).
        
        links: n-vector, with values in {0,...,n-1}. These are the intercustomer
        links, as described in the paper. 
        """
        return self.logP[self.ii,links].sum()        
    
    
    def sample(self):
        P, n = self.P, self.n
        return np.array([choice(n,p=P[i,:]) for i in range(n)])
    
    def samples(self,m):
        P, n = self.P, self.n
        return np.array([choice(n,m,p=P[i,:]) for i in range(n)])



    
    def sample_gumbel(self):
        logP, n = self.logP, self.n
        G = -np.log(-np.log(rand(n,n)))
        G += logP
        return np.argmax(G,axis=1)
    
    
    
class CRP(DDCRP):
    """
    Traditional one-parameter CRP, implemented as DDCRP
    """
    def __init__(self,n,alpha):
        P = np.tril(np.full((n,n),1.0))
        np.fill_diagonal(P,alpha)
        super().__init__(P)
        
        


        
        
def links2labels(links):
    """
    Computes the limited growth string that describes the partition implied
    by the given intercustomer links.
    """
    n = len(links)
    labels = np.full(n,-1)
    label = -1
    for i,j in enumerate(links):
        if labels[j] < 0: 
            label += 1
            labels[j] = label
        labels[i] = labels[j]    
    return labels    
    
        
        
        
    
    
    