import numpy as np
from numpy.random import randn
#from autograd import grad


def cs(f,x):
    """Applies complex-step to get value and 1st derivative of f at x.
    
    Applicable to: f(scalar) 
                   f(ndarray) where f is evaluated elementwise
                   
    returns: f(x) and a function back, such that back(dy) = df(x)/dx * dy               
    
    """
    z = f(x+1e-20j)
    dydx = 1e20*np.imag(z) 
    return np.real(z), lambda dy: dy * dydx




def rdot(x,y):
    """Recursive inner product applied to two tuples of identical structure.
    
    The tuple elements can be scalars, ndarrays, or tuples (of tuples of .... ) 
    of scalars and ndarrays.
    
    Elelements can be real or complex. No complex conjugation is done.
    
    returns: the scalar inner (i.e. dot) product = sum of all products of pairs
             of scalars retrieved in parallel from the inputs.
             
    assertions will fail if tuple structures are incompatible.         
    
    """
    if np.isscalar(x):
        assert np.isscalar(y)
        return x*y
    if isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray)
        assert x.shape == y.shape
        return (x*y).sum()
    return sum(rdot(xi,yi) for xi,yi in zip(x,y))
    
def raug(x,r,eps=1e-20j):
    """Recursive complex tuple augmentation.
    
    returns: x + r*eps, packed in a tuple of the same structure as x and r.
    """
    if np.isscalar(x):
        assert np.isscalar(r)
        return x + r*eps
    if isinstance(x, np.ndarray):
        assert isinstance(r, np.ndarray)
        assert x.shape == r.shape
        return x + r*eps
    return tuple( raug(xi,ri,eps) for xi,ri in zip(x,r) )
    
def rprod(x,r):
    """Recursive complex tuple augmentation.
    
    returns: x + r*eps, packed in a tuple of the same structure as x and r.
    """
    if np.isscalar(x):
        assert np.isscalar(r)
        return x * r
    if isinstance(x, np.ndarray):
        assert isinstance(r, np.ndarray)
        assert x.shape == r.shape
        return x * r
    return tuple( rprod(xi,ri) for xi,ri in zip(x,r) )




def rrandn(x):
    """Recursively builds a tuple filled with randn elements, of the
    same structure as x.
    """
    if np.isscalar(x):
        return randn()
    if isinstance(x, np.ndarray):
        return randn(*x.shape)
    return tuple(rrandn(xi) for xi in x)

def rimag(x,scale=1e20):
    """Recursively extracts the scaled imaginary part from tuple elements. 
    Returns a tuple of the same structure as x.
    """
    if np.isscalar(x) or isinstance(x, np.ndarray):
        return scale*np.imag(x)
    return tuple(rimag(xi,scale) for xi in x)


def astuple(x):
    """Returns x or (x,)"""
    return x if isinstance(x,tuple) else (x,)


def cstest(f,*x,**kwargs):
    """
    Tests the backpropagation of fback against forward-mode complex-step
    differentiation applied to f.
    
    parameters: 
        
        f: a function of one or more inputs, returning one or more outputs.
        
        fback: a version of f that returns the same output(s) as f and also
               a backpropagation handle that takes as many inputs as f 
               has outputs---and outputs as many values as f has inputs. 
               
         *x: An argument list to be passed to f and fback. The derivatives are 
         tested only at *x.
             
    returns: abs of : rx' J' ry - ry' J rx, where rx and ry are randn-generated
                      with the same (effective) dimensions as (respectively) all 
                      inputs and all outputs. J'ry is computed via back(ry), while
                      J rx is computed by the complex-step. 
    
    
    """
    *y, back = f(*x,deriv=True,**kwargs)    # y is a list of one or more return values from  f
    ry = rrandn(tuple(y))
    dx = astuple(back(*ry))  # dx = J' ry
    assert len(dx) == len(x)

    rx = rrandn(x)
    cx = raug(x,rx)
    dy = rimag(astuple(f(*cx,**kwargs))) # dy = J rx
    
    return abs( rdot(ry,dy) - rdot(rx,dx) )
    
def cstest_w(f,x,w,**kwargs):
    """
    
    """
    w = tuple(xi*wi for wi,xi in zip(w,x))
    
    *y, back = f(*x,deriv=True,**kwargs)    # y is a list of one or more return values from  f
    ry = rrandn(tuple(y))
    dx = astuple(back(*ry))  # dx = J' ry
    assert len(dx) == len(x)

    rx = rprod(rrandn(x),w)
    cx = raug(x,rx)
    dy = rimag(astuple(f(*cx,**kwargs))) # dy = J rx
    
    return abs( rdot(ry,dy) - rdot(rx,dx) )






def optobjective(f,trans,sign=1.0,**kwargs):
    """Wrapper for f to turn it into a minimization objective.
    
    The objective is: 
        obj(x) = sign*f(*trans(x),**kwargs) 
    where f can take multiple inputs and has a scalar output. 
    The input x is a vector.
    
    Both f(...,deriv=True) and trans(...) return backpropagation 
    handles, but obj(x) immediately returns value, gradient.  
    
       
    """
    def obj(x):
        *args, back1 = trans(x, deriv=True)
        y, back2 = f(*args,deriv=True,**kwargs)
        g = back1(*back2(sign))
        return sign*y, g
    
    return obj
    
def csHvp(obj,x,v,eps=1e-20, test = False):
    """Returns H @ v, where v is a vector and H is the Hessian of obj(x), using
    complex step differentiation and the Pearlmutter trick.
    
    Performs one complex-step, forward-mode differentiation by calling
    obj(x + eps*1.0j*v).
    
    The objective function, obj maps R^n -> R. It must return the function value 
    and the gradient. (See optobjective.) The gradient of obj cannot be done by
    complex step differentiation, because we cannot nest the complex step trick.
    
    If test = True, it checks the gradient returned by obj against the 
    complex-step gradient. This check does not require another function 
    evaluation and happens almost for free. If negative, the test fails with an 
    assertion.
    
    
    """
    y, cg = obj(x+eps*1.0j*v)
    if test:
        dy = np.imag(y)/eps
        g = np.real(cg)
        assert abs(g@v - dy) < np.sqrt( eps * (g@g) * (v@v) )
    return np.imag(cg)/eps
    
def csHess(obj,x,I,eps=1e-20, test = False):
    """
    Computes the full Hessian of obj at x, using len(x) calls to
    csHvp and therefore len(x) function evaluations. The caller must supply 
    I = np.eye(len(x)). See csHvp for details abouut the other parameters.     
    
    
    """
    n = len(x)
    H = np.empty((n,n))
    for i in range(n):
        H[:,i] = csHvp(obj,x,I[:,i],eps,test)
    return H    
        

def cstestobj(obj,x,eps=1e-20,n = 5):
    """
    Tests objective function, obj gradient using n complex-step, forward-mode
    differentiations at n random (vector-valued) input perturbations. Returns
    the maximum discrepancy over the n tests.
    """
    R = randn(len(x),n)
    mx = 0
    for i in range(n):
        ri = R[:,i]
        y, cg = obj(x+eps*1.0j*ri)
        g = np.real(cg)
        dy = np.imag(y)/eps  # dy = g ' * ri
        mx = np.maximum(mx,abs(dy-g@ri))
    return mx
    


if __name__ == "__main__":
    print("Running test script for module adtools\n")

    
    
    print('\nApplying cstest to prodfun:')
    def prodfun(X,Y,deriv=False):
        P = X @ Y
        if not deriv: return P
        return P, lambda dP: (dP @ Y.T, X.T @ dP)
    
    
    m,k,n = 2,3,4
    X = randn(m,k)
    Y = randn(k,n)
    delta = cstest(prodfun,X,Y)
    print(delta)
        
    
    
    
    
    
    
    
    
    
    
    
    


