"""
Redefines some NIG functions to produce matrix outputs, for data vectors AND 
vectors of parameters.


"""

import numpy as np
from numpy.random import randn

from lib.special.besselk import logK1e
from lib.deriv.adtools import cstest, cstest_w
from lib.deriv.broadcast_adtools import sum2shape

logpi = np.log(np.pi)


def asrows(x):
    if np.isscalar(x): return np.full((1,1),x), 0, (1,1)
    if x.ndim==1: 
        x = x.reshape(1,-1)
        return x, 1, x.shape
    assert x.ndim==2
    return x, 2, x.shape

def ascols(x):
    if np.isscalar(x): return np.full((1,1),x), 0, (1,1)
    if x.ndim==1: 
        x = x.reshape(-1,1)
        return x, 1, x.shape
    assert x.ndim==2
    return x, 2, x.shape

def unshape(x,ndim):
    if ndim==0: return x.item()
    if ndim==1: return x.reshape(-1)
    assert ndim==2
    return x




def niglogpdf_mat(x,b,gamma, deriv = False):
    """If gamma and b are m-vectors and x is an n-vector, then an m-by-n 
    matrix of logpdf values will be returned. 
    
    """    
    assert np.shape(b) == np.shape(gamma)
    x, xdim, xshape = asrows(x)    
    b, bdim, bshape = ascols(b)
    gamma, dummy1, dummy2 = ascols(gamma)
    
    g2b2 = gamma**2 + b**2
    loga = np.log(g2b2) / 2
    
    c = loga + gamma - logpi
    logq = np.log1p(x**2)/2.0  
    aq = np.exp(loga + logq)                  # broadcasting
    
    if not deriv:
        return c + logK1e(aq) + b*x - aq - logq  
    
    logk1e, back1 = logK1e(aq, deriv = True)
    y = c + logk1e + b*x - aq - logq          # broadcasting

    def back(dy):
        dc = sum2shape(dy,bshape)   
        dx, db = sum2shape(b*dy,xshape), sum2shape(x*dy,bshape)       
        daq = back1(dy) - dy              
        dlogq = sum2shape(-dy,xshape)
        
        daqaq = daq*aq
        dloga = sum2shape(daqaq,bshape)            
        dlogq += sum2shape(daqaq,xshape) 
        
        dx += x*dlogq/(1+x**2)
        
        dloga += dc                       
        dgamma = dc                       
                
        dgamma += dloga*gamma/g2b2        
        db += dloga*b/g2b2       

        
        return unshape(dx,xdim), unshape(db,bdim), unshape(dgamma,bdim)
        
    return y, back 



def fullniglogpdf_mat(logscale,location,b,gamma,*,data,acc=False,deriv=False):
    return scaled_shifted_logpdf_mat(niglogpdf_mat, logscale, location, b, gamma,
                                 data=data, deriv=deriv)





def scaled_shifted_logpdf_mat(logpdf,logscale,shift,*shape,
                          # must be passed with keyword: 
                          data,    
                          deriv = False):
    
    x = data
    
    assert np.ndim(x) < 2 # scalar or vector
    assert np.ndim(logscale) < 2 # scalar or vector
    p_shape = np.shape(logscale)
    assert p_shape == np.shape(shift)
    
    if not np.isscalar(logscale):
        logscale = logscale.reshape(-1,1)              
        shift = shift.reshape(-1,1)       
    col_shape = np.shape(logscale)
    
    
    scale = np.exp(logscale)
    z = (x-shift)/scale                          # broadcast
    
    if not deriv:
        y0 = logpdf(z,*shape)
        return y0 - logscale             
            
    y0, back1 = logpdf(z,*shape, deriv=True)
    y = y0 - logscale                            # broadcast 
    
    def back(dy):
        
        dlogscale = sum2shape(-dy,col_shape)
        dz, *dshape = back1(dy)
        
        dshift = sum2shape(-dz,col_shape) / scale
        dscale = sum2shape(-z*dz,col_shape) / scale
        dlogscale += scale*dscale
        
        if not np.isscalar(logscale):
            dlogscale = dlogscale.reshape(p_shape)              
            dshift = dshift.reshape(p_shape)       
        
        
        
        return (dlogscale, dshift,*dshape) 
    
    
    return y, back


if __name__ == "__main__":
    print("Running test script for module nig_mat\n")



    print("\nTesting backpropagation of niglogpdf_mat:")
    m,n = 2, 1
    x = randn(n)
    b = randn(m)
    gamma = randn(m)**2
#    w = [1,1,1]
#    delta = cstest_w(niglogpdf_mat,(x,b,gamma),w)
#    print(delta)    


    y0 = niglogpdf_mat(x,b,gamma)
    print("y0: ",y0)

    y1, back = niglogpdf_mat(x,b,gamma,deriv=True)
    print("y1: ",y1)


    
#    print("\nTesting backpropagation of fullniglogpdf_mat:")
#    logscale, shift = randn(m), randn(m)
#    delta = cstest(fullniglogpdf_mat,logscale,shift,b,gamma,data=x)
#    print(delta)    
    


        

