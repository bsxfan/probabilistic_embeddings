"""
See Odyssey 2020 paper: https://arxiv.org/abs/2004.04096
"""


import numpy as np

class model:
    
    
    def __init__(self,w):
        """
        Constructs diagonal 2covariance PLDA model, from given model parameters.
        
        w: d-vector of within-class precisions. (Between-class variances are 
           assumed unity.) The entries of w must be strictly positive.
        
        
        """
        self.w = w
        self.d = len(w)
        
        
    def logLH(self,S,X,B=None):
        """
        Computes log-likelihoods: log P(observations| cluster) + const,
        for a number of given clusters. The observations are represented by 
        X,B and the clusters by S.
        
        S: (m,n) matrix, with 0/1 entries, where m is the number of clusters
           and n the total number of observations in X,B. Observation j is in
           cluster i if S[i,j]. The clusters need not be mutually non-
           overlapping, so that logLH may be used to compute likelihoods for
           competing hypotheses.
            
        X: (n,d) matrix. Contains embedding point-estimates;
           n of them, of dimension d.
        
        B: (n,d) matrix. Contains embedding estimate precisions;
           n of them, of dimension d.
           Optional: if B is None, or not given it is assumed to be infinite.
           
        
        
        example: When n = 2 and S = np.array([[1,1], [1,0], [0,1]]), then
                 logLR = np.array([1,-1,-1]) @ model.logLH(S,X,B) is the logLR 
                 score for the usual 2-input speaker verification trial.
                 
        example: When S is a large sparse matrix, a whole database of 
                 verification trials can be scored in one call logLH, followed 
                 by another linear operation.
                 
        example: When S indexes all possible subsets of n inputs, then a single 
                 call to logLH, followed by another linear operation can
                 compute the log-likelihoods for each of the partitions of
                 a set of n inputs.
        
        """
        w = self.w
        if B is None:
            n,d = X.shape
            R = np.tile(w,(n,1))
        else: 
            wB = w*B
            R = wB / (w+B)
        RX = R*X         #(n,d)
        
        SRX = S @ RX     #(m,d)
        SR = S @ R       #(m,d)
        
        LLH = (SRX**2/(1.0+SR) - np.log1p(SR) ).sum(axis=1) / 2.0   # m-vector
        
        return LLH
        
        
        