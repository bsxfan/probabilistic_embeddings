import numpy as np
from numpy.random import randn

from lib.prob.crp import CRP
from ntuplemodel import NTupleModel
from diag2covplda import model as PLDA
import lib.combin.partitions as part
from lib.deriv.adfunctions import logsumexp


dim = 10
w0 = 5.0
b0 = 5.0
w = np.full(dim,w0)    # high precision (low noise)
plda = PLDA(w)

alpha, beta = 1, 0.5
crp = CRP(alpha,beta)

n = 3
tmodel = NTupleModel(n,crp,plda)

for i in range(10):
    rgs, counts = crp.sample(n)
    print(rgs,' --> ',tmodel.index(rgs))
    
    m = len(counts)
    Y = randn(m,dim)
    X = Y[rgs,:] + randn(n,dim) / np.sqrt(w)
    B = np.full((n,dim),b0)

    lp = tmodel.logposterior(X,B)
    post = np.exp(lp - logsumexp(lp))
    print(post,'\n')    