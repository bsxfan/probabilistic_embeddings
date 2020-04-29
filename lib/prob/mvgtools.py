"""
Tools for multivariate Gaussians
"""

import numpy as np
from numpy.random import randn, randint
from lib.matrix import CholMat, one_hot, sum_of_columns, addtodiag, \
                        inplace_trmm, frandn 




def natparams_to_mvgstats(Q:CholMat, Qmu:np.ndarray):
    """Computes first and 2nd order stats from Gaussian natural parameters.
    
    Parameters:
        Q: CholMat, the Cholesky factored precision matrix. 
           Note: precision = Q.L @ Q.R; covariance = Q.R.I @ Q.L.I 
       
        Qmu: numpy vector, precision @ mean
        
    Returns:
        mu: numpy array vector, mean = <x>
        S:  square numpy array, <xx'> = covariance + mu @ mu' 
    
    """
    Li, Ri = Q.L.I, Q.R.I  # these are the inverses of the Cholesky factors (not explicit)
    z = Li @ Qmu  # invokes solve
    mu = Ri @ z   # invokes solve
    S = Ri @ addtodiag(np.outer(z,z),1) @ Li # invokes 2 solves
    return mu, S




class MVG:
    """Base class for multivariate Gaussians. Contains only dimensionality."""
    
    def __init__(self,dim):
        self.dim = dim
        self.dimlog2pi = dim*np.log(2*np.pi) # subtract 1/2 this to get normalized log pdf
        
        
    def logpdf(self,X, deriv=False):
        """Evaluates log PDF on each column of X""" 
        if deriv:
            y, back = self.loglikelihood(X, deriv=True)
        else:
            y = self.loglikelihood(X)
        y += (self.logdetQ - self.dimlog2pi)/2 
        if deriv: return y, back 
        else: return y
        
    
    
def loglikelihood_cs(X, *, QR, mu, deriv=False):
    """Complex-step friendly version of MVG_mu.loglikelihood, with optional 
       backpropagation handle.
    
       X: data vectors are in columns
       QR: TrMatrix: RHS factor of precision Q, so that QR.T @ QR = Q  
       deriv: if True, also returns handle to backpropagate into X (not QR or mu)
       
       X may be complex, but not QR or mu. This facilitates complex step
       verification of the first derivative, or complex step computation of 
       Hessian-vector products.
       
    """
    Z = QR @ (X - mu)  
    y = -0.5 * (Z**2).sum(0)
        
    if not deriv: return y
    def back(dy): return (-dy) * (QR.T @ Z)     
    return y, back    
        


      
        
class MVG_mu(MVG):
    """Base class for multivariate Gaussian. Contains only mean"""
    
    def __init__(self,mu):
        super().__init__(len(mu))
        self.mu = mu
        self.mucol = mu.reshape(-1,1)
        self.murow = mu.reshape(1,-1)
        
    def loglikelihood(self,X, deriv = False):
        """Evaluates unormalized Gaussian on each column of X"""
        
        if deriv == True or np.iscomplexobj(X): 
            return loglikelihood_cs(X, QR=self.Q.R, mu=self.mucol, deriv=deriv)
        
        # We can do the calculation without this allocation, but the 
        # explicit Delta is numerically more stable if the data is far from the
        # mean.
        Delta = X - self.mucol  
        Z = inplace_trmm(self.Q.R, Delta)
        Z **= 2  # square inplace
        z = Z.sum(0)
        z *= (-0.5)
        return z
        
        
    def sample(self,n):
        """Samples n draws into a dim-by-n matrix."""
        X = inplace_trmm(self.cholC.L, frandn(self.dim,n))
        X += self.mucol
        return X
        
    def sample_and_score(self,n):
        """Samples n draws into a dim-by-n matrix. Also returns a vector of
        the logPDF values of these draws."""
        
        Z = frandn(self.dim,n)
        ZZ = Z**2
        logPDF = ZZ.sum(0)
        logPDF *= (-0.5)
        logPDF += (self.logdetQ - self.dimlog2pi) / 2

        Z = inplace_trmm(self.cholC.L, Z)
        Z += self.mucol
        
        

        return Z, logPDF
        
        
class MVG_Pmu(MVG_mu):
    """MVG defined in terms of precision and mean."""
    
    def __init__(self,Q,Qmu):
        """
        Construct Gaussian from natural parameters:
            Q: precision, positive definite ndarray, or CholMat
            Qmu: precision @ mean
        """

        if isinstance(Q,np.ndarray):
            Q = CholMat(Q)
        assert isinstance(Q,CholMat)
        self.Q = Q    
        self.logdetQ = Q.logdet()
            
        self.cholC = Q.I   # Cholesky factored covariance CholMatI (inverse not explicitly computed)
        mu, self.S = natparams_to_mvgstats(Q, Qmu)
        super().__init__(mu)
        
        
        

class MVG_Cmu(MVG_mu):
    """MVG defined in terms of covariance and mean."""
    
    def __init__(self,C,mu):
        """
        Construct Gaussian from mean and covariances:
            C: positive definite ndarray
            mu: mean
        """
        super().__init__(mu)
        self.C = C
        self.cholC = CholMat(C)

        self.Q = self.cholC.I
        self.logdetQ = self.Q.logdet()

        self.S = C + np.outer(mu,mu)
        
        
if __name__ == "__main__":
    print("Running test script for module mvgtools\n")

    dim, n = 100, 10000
    R = randn(dim,2*dim)
    C = (R @ R.T ) / (2*dim)
    mu = randn(dim)
    
    mvg = MVG_Cmu(C,mu)
    X = mvg.sample(n)
    mvg1 = MVG_Cmu(2*C,mu)
    mvg2 = MVG_Cmu(C/2,mu)
    mvg3 = MVG_Cmu(C,mu+randn(dim))


    logPX = mvg.logpdf(X).sum()
    logPX1 = mvg1.logpdf(X).sum()
    logPX2 = mvg2.logpdf(X).sum()
    logPX3 = mvg3.logpdf(X).sum()
    print("These should be positive:\n",logPX - np.array([logPX1, logPX2, logPX3]))

    Q = np.linalg.inv(C)
    Qmu = Q @ mu
    mvgQ = MVG_Pmu(Q,Qmu)
    logPXQ = mvgQ.logpdf(X).sum()
    print("This should be close to zero:\n",(logPX-logPXQ)/logPX)


    print("\nTesting stats:")

    dim, n = 4, 1000000
    R = randn(dim,2*dim)
    C = (R @ R.T ) / (2*dim)
    mu = randn(dim)
    
    mvg = MVG_Cmu(C,mu)
    X = mvg.sample(n)

    Q = np.linalg.inv(C)
    Qmu = Q @ mu
    mvgQ = MVG_Pmu(Q,Qmu)

    S = (X @ X.T) / n
    print("data S = \n",S)
    print("Pmu S - Cmu S= \n",mvgQ.S - mvg.S)
    print("Cmu S - data S= \n",mvg.S - S)
    print("Pmu S - data S= \n",mvgQ.S - S)


    print("\nTesting sample and score:")
    
    from scipy import stats
    
    A = randn(3,4); C = A @ A.T
    mu = randn(3)
    d = MVG_Cmu(C,mu)
    ds = stats.multivariate_normal(mean=mu,cov=C)
    X, y = d.sample_and_score(5) 
    print(abs(y-d.logpdf(X)).max())
    print(abs(y-ds.logpdf(X.T)).max())


    from lib.deriv.adtools import cstest
    print("\nTesting logpdf backprop:")
    delta = cstest(d.logpdf,X)
    #delta = cstest(loglikelihood_cs, X, mu=d.mucol, QR = d.Q.R)
    print(delta)
    
        
        
        