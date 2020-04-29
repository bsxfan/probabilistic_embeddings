import numpy as np
from numpy.random import randn

from lib.prob.crp import CRP
#from diag2covplda import model as PLDA
#import lib.combin.partitions as part
#from lib.deriv.adfunctions import logsumexp

from ahc import AHC

dim = 10
w0 = 50.0
b0 = 50.0
w = np.full(dim,w0)    # high precision (low noise)
#plda = PLDA(w)

alpha, beta = 1, 0.5
n = 10
crp = CRP(alpha,beta)
print('prior = ',crp,':')
print('E{#tables} for ',n,'customers = ',crp.exp_num_tables(n))

rgs, counts = crp.sample(n)
print('prior sample:', rgs)

m = len(counts)
Y = randn(m,dim)
X = Y[rgs,:] + randn(n,dim) / np.sqrt(w)
B = np.full((n,dim),b0)

        
        
ahc = AHC(X,B,w,crp)
ll = ahc.cluster()
print('ahc result: ',ll)
