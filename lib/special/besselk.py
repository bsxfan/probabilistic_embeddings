"""
Module providing convenient and fast derivatives of log(scipy.special.k1e)

See the function: logK1e


Asymptotes:
    Kn(x) ~ sqrt(pi/2x) exp(-x) as x -> inf
    
    Kn(x) ~ 1/2 Gamma(n) / (1/2 x)^n, as x -> 0, for n > 0
    K0(x) ~ -log(x), as x -> 0
    
    


"""

import scipy.special as special
import numpy as np
from lib.deriv.adtools import cs



def k012e(x):
    """Returns k0e(x), k1e(x) and k2e(x) for real or complex x.
    
    For real x, the fast exponentially scaled K_n Bessel functions, k0e end k1e 
    are defined in scipy.special, but not ke2, which is computed here from the 
    other two using the recurion: 
        K_n(x) = K_{n-2}(x) + 2(n-1)/z * K_{n-1}(x) 
    
    For complex x, kve(0,x) and kve(1,x) and the recursion are used.
    
    Returns the outputs all three functions, evaluated at x.
    """
    if np.iscomplexobj(x):
        k0 = special.kve(0,x)
        k1 = special.kve(1,x)
        k2 = k0 + 2.0*k1/x
    else:
        k0 = special.k0e(x)
        k1 = special.k1e(x)
        k2 = k0 + 2.0*k1/x
    return k0, k1, k2

#def ddxlogK1e(z):
    """
    ddx K1(z) = (-1/2) * ( K0(z) + K2(z) )
    ddx log(exp(z)K1(z)) 
    = 1 + [ddx K1(z)] / K1(z)
    = 1 - [K0(z) + K2(z)] / [2*K1(z)]
    
    """    


def k0e(x):
    if np.iscomplexobj(x):
        return special.kve(0,x)
    else:
        return special.k0e(x)

def k1e(x):
    if np.iscomplexobj(x):
        return special.kve(1,x)
    else:
        return special.k1e(x)



class csret():
    def __init__(self,z):
        self.val = np.real(z)
        self.deriv = 1e20*np.imag(z)
    
def cs2(f,x):
    z1, back = f(x+1e-20j)
    z2 = back(1.0)
    return csret(z1), csret(z2)





def logK1e(x, deriv = False, complex_step = True):
    """
    log(scipy.special.k1e), with derivative capabilities.
    
    k1e(x) is Bessel function K1(x), scaled by exp(x)  
    
    parameters:
        
        x: real or complex ndarray
        deriv: Bool, optional, default = False
               - False: return only function values
               - True: return function values and a backpropagation function.
        
        - When complex_step=True and x is real, the first derivative is computed 
          with complex step differentiation. This is faster.
        - Otherwise, an explicit first derivative calculation is used.

    
    The whole function can be wrapped in a complex step differentiation. Then
    the function value, the complex-step first derivative, the explicit first
    derivative and the second derivative can all be recovered from the real and
    imaginary parts of the two return values.
    
    
    """
    
    
    complexx = np.iscomplexobj(x)
    complex_step  = complex_step and not complexx
        
    if not deriv: return np.log(k1e(x))

    if complex_step:
        y, dx = cs(k1e,x)
        return np.log(y), lambda dy: dx(dy/y)

    k0 = k0e(x)
    k1 = k1e(x)
    k2 = k0 + 2.0*k1/x
    ddx =  1 - (k0 + k2) / (2*k1)
    return np.log(k1), lambda dy: dy * ddx 



def logK1e_2ndderiv(x):
    """Slow and probably inaccurate for large x. For testing only."""
    f = special.k1e(x)
    k1 = special.kvp(1,x,1)
    k2 = special.kvp(1,x,2)
    e = np.exp(x)
    f1 = f + e*k1
    f2 = f1 + e*(k1+k2)
    return (f2*f - f1**2) / f**2


if __name__ == "__main__":
    print("Running test script for module besselk\n")
    
    from numpy.random import randn


#    n = 10
#    m = 3
#    x = randn(n,m)**2

    x = randn(2,1)**2
    
    y0 = logK1e(x)
    y1, back = cs(logK1e,x); ddx1 = back(1.0)
    y2, back = logK1e(x, deriv = True); ddx2 = back(1.0)  
    y3, back = logK1e(x, deriv = True, complex_step = False); ddx3 = back(1.0)  
    
    r1, r2 = cs2(lambda x: logK1e(x,True), x)
    y4, ddx4 = r1.val, r1.deriv
    ddx5, d2dx5 = r2.val, r2.deriv
    
    d2dx6 = logK1e_2ndderiv(x)
    
    print('comparing values:')
    for yi in (y1,y2,y3,y4):
        print(abs(y0-yi).max())

    print('\ncomparing derivatives:')
    for ddxi in (ddx2,ddx3,ddx4,ddx5):
        print(abs(ddx1-ddxi).max())

    print('\ncomparing 2nd derivatives:')
    print(abs(d2dx5-d2dx6).max())

                





    
    