"""
NIG is the normal inverse Gaussian distribbution. 

See: - https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution
     - Dimitris Karlis. An EM type algorithm for maximum likelihood estimation 
       of the normal–inverse Gaussian distribution.
       
The parametrizations in those two references (Wikipedia, Karlis), specified 
in terms of (alpha, beta, mu, delta) are equivalent. 

To translate to the scipy NIG definition, set mu = 0 and delta = 1. Then
a = alpha and b = beta. The scipy definition is a standard version, 
parametrized with only (a,b), but it can then be generalized by location and 
scale transforms, just like the other scipy.stats continuous distributions.         
   
There is a derived parameter: gamma = sqrt(a**2 - b**2). 
    
For the standard scipy NIG(a,b):
    mean    : mu0 = b/gamma 
    variance: v0  = a**2 / gamma**3  
                  = (b**2 + gamma**2) / gamma**3
                  = (mu0**2 + 1) / gamma
             
For scipy NIG(a,b,loc = mu1, scale = s):
    mean    : mu1 + s * mu0
    variance: s**2 * v0

If x ~ NIG(a,b), then y = mu1 + s* x  ~ NIG(a,b,loc=mu1,scale = s)
                
              
The NIG functions below are parametrized with (log(scale), loc, b, gamma)              
                           
"""


import scipy
import scipy.stats as stats
#import scipy.special as special

import numpy as np
from numpy.random import randn, rand

from lib.special.besselk import logK1e
from lib.deriv.adtools import cstest





logpi = np.log(np.pi)

def niglogpdf(x,b,gamma, acc = False, deriv = False):
        
    n = np.size(x)
    
    
    g2b2 = gamma**2 + b**2
    loga = np.log(g2b2) / 2
    
    c = loga + gamma - logpi
    logq = np.log1p(x**2)/2.0  
    aq = np.exp(loga + logq)                  # broadcasting
    
    if not deriv:
        y = c + logK1e(aq) + b*x - aq - logq  
        return np.sum(y) if acc else y
    
    logk1e, back1 = logK1e(aq, deriv = True)
    y0 = c + logk1e + b*x - aq - logq         # broadcasting
    y = np.sum(y0) if acc else y0

    def back(dy):
        dc = n*dy if acc else np.sum(dy)   
        dx, db = b*dy, np.sum(x*dy)       
        daq = back1(dy) - dy              
        dlogq = -dy
        
        daqaq = daq*aq
        dloga = np.sum(daqaq)            
        dlogq += daqaq 
        
        dx += x*dlogq/(1+x**2)
        
        dloga += dc                       
        dgamma = dc                       
        
        
        dgamma += dloga*gamma/g2b2        
        db += dloga*b/g2b2                
        
        return dx, db, dgamma
        
    return y, back 
        




def fullniglogpdf(logscale,location,b,gamma,*,data,acc=False,deriv=False):
    return scaled_shifted_logpdf(niglogpdf, logscale, location, b, gamma,
                                 data=data, acc=acc, deriv=deriv)

        
def to_scipy_distr(logscale,location,b,gamma):
    a = np.sqrt(gamma**2 + b**2)
    scale = np.exp(logscale)
    return stats.norminvgauss(a,b,loc=location,scale=scale)    
    
        
def scaled_shifted_logpdf(logpdf,logscale,shift,*shape,
                          # must be passed with keyword: 
                          data,    
                          deriv = False, 
                          acc = False ):
    
    x = data
    n = np.size(x)
    scale = np.exp(logscale)
    z = (x-shift)/scale
    
    if not deriv:
        y0 = logpdf(z,*shape, acc)
        return y0 - (n*logscale if acc else logscale)
            
    y0, back1 = logpdf(z,*shape, acc=acc, deriv=True)
    y = y0 - (n*logscale if acc else logscale)
    
    def back(dy):
        dlogscale = -n*dy if acc else -np.sum(dy)
        dz, *dshape = back1(dy)
        dshift = -(dz.sum())/scale
        dscale = -np.sum(z*dz)/scale
        dlogscale += scale*dscale
        return (dlogscale, dshift,*dshape) 
    
    
    return y, back
        










            
if __name__ == "__main__":
    print("Running test script for module nig\n")
            
            
    
    n = 5  # does work for n=1
    x = randn(n)  # also works for x = randn(), or x = randn(m,n)
    b = randn()
    gamma = randn()**2
    #a = np.sqrt(gamma**2 + b**2)
    
    
    print("Comparing niglogpdf vs stats.norminvgauss.logpdf:")
    y1 = niglogpdf(x,b,gamma)
    y2 = to_scipy_distr(0,0,b,gamma).logpdf(x)
    #y2 = stats.norminvgauss.logpdf(x,a,b)
    print(abs(y1-y2).max())

    print("\nComparing fullniglogpdf vs stats.norminvgauss.logpdf:")
    scale = randn()**2
    mu = randn()
    y1 = fullniglogpdf(np.log(scale),mu,b,gamma,data=x)
    y2 = to_scipy_distr(np.log(scale),mu,b,gamma).logpdf(x)
    #y2 = stats.norminvgauss.logpdf(x,a,b,loc=mu,scale=scale)
    print(abs(y1-y2).max())

    
    print("\nTesting backpropagation of niglogpdf(..., acc = False):")
    #f, fback, args = editfun(niglogpdf,x,b,gamma, flags = [1,1,1] )
    
    delta = cstest(niglogpdf,x,b,gamma)
    print(delta)    

    print("\nTesting backpropagation of niglogpdf(..., acc = True):")
    delta = cstest(niglogpdf,x,b,gamma,acc=True)
    print(delta)    

    print("\nTesting backpropagation of fullniglogpdf(..., acc = False):")
    logscale, shift = randn(), randn()
    delta = cstest(fullniglogpdf,logscale,shift,b,gamma,data=x)
    print(delta)    

    print("\nTesting backpropagation of fullniglogpdf(..., acc = True):")
    delta = cstest(fullniglogpdf,logscale,shift,b,gamma,data=x,acc=True)
    print(delta)    


