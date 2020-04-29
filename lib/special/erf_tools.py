import numpy as np 
from scipy import special

sqrtpi = np.sqrt(np.pi)
sqrt2 = np.sqrt(2)
ln10 = np.log(10)
log2onsqrtpi = np.log(2.0/sqrtpi)
log2sqrt2 = np.log(2*sqrt2)
ln2 = np.log(2)


def erf(x,deriv = False):
    
    y = special.erf(x)
    if not deriv: return y
    
    return y, lambda dy: (2.0/sqrtpi)*dy*np.exp(-x**2)
    

def erf_and_logderiv(x, deriv = False):
    
    y = special.erf(x)
    logderiv = log2onsqrtpi - x**2
    
    if not deriv: return y, logderiv

    def back(dy, dlogderiv): 
        return dy*np.exp(logderiv) - 2*dlogderiv*x
        
    return y, logderiv, back






def cs_erfinv(p):
    """Complex-step friendly version of scipy.special.erfinv. 
    
    For complex p, it returns the complex number:
        
        x + j*dx
        
    where x = erfinv(real(p)) and dx = dx/dp * imag(p)     
    """
    if not np.iscomplexobj(p):
        return special.erfinv(p)
    rp = np.real(p)
    x = special.erfinv(rp)
    dp = (sqrtpi/2.0)*np.exp(x**2)*np.imag(p)
    return x + 1.0j*dp
    
    
    
    
def erfinv(p,deriv = False):
    
    if not deriv: return cs_erfinv(p)
    
    x = cs_erfinv(p)
    back = lambda dy: (sqrtpi/2.0)*dy*np.exp(x**2)
    return x, back


def Phi(x,deriv = False):
    """Normal cumulative distribution."""
    arg = x/sqrt2
    if not deriv: return (1 + erf(arg) ) / 2.0
    
    e, back1 = erf(arg,deriv=True)
    y = (1+e)/2
    def back(dy): return back1(dy/2.0)/sqrt2
    return y, back

def Phi_and_logderiv(x,deriv = False):
    
    arg = x/sqrt2
    #if deriv: e, logderiv1, back1 = erf_and_logderiv(arg, deriv = deriv)
    e, logderiv1 = erf_and_logderiv(arg)
    y = (1+e)/2.0
    logderiv = logderiv1 - log2sqrt2
    if not deriv: return y, logderiv

    def back(dy, dlogderiv): 
        return dy*np.exp(logderiv) -sqrt2*dlogderiv*arg
        #return back1(dy/2.0,dlogderiv)/sqrt2

    return y, logderiv, back



def Phi_inv(p,deriv = False):

    if not deriv: return sqrt2*erfinv(2.0*p - 1.0)

    arg, back1 = erfinv(2.0*p - 1.0, deriv=True)
    x = sqrt2*arg
    def back(dx): return 2.0*back1(sqrt2*dx)
    return x, back

probit = Phi_inv


def dprime2EER(dprime,deriv = False):
    
    if not deriv: return Phi(-0.5*dprime)
    
    EER, back1 = Phi(-0.5*dprime,deriv=True)
    def back(dEER): return -0.5*back1(dEER)
    return EER, back


def logdprime2EER(logdprime,deriv = False):
    
    arg = -0.5*np.exp(logdprime)
    if not deriv: return Phi(arg)
    
    EER, back1 = Phi(arg,deriv=True)
    def back(dEER): return arg*back1(dEER)
    return EER, back


def logdprime2EER_and_logabsderiv(logdprime,deriv = False):
    
    arg = -0.5*np.exp(logdprime)
    if not deriv: EER, logderiv1 = Phi_and_logderiv(arg)
    else: EER, logderiv1, back1 = Phi_and_logderiv(arg, deriv = True)
    logderiv = logderiv1 + logdprime - ln2
    if not deriv: return EER, logderiv 

    def back(dEER, dlogderiv): 
        return dlogderiv + arg*back1(dEER,dlogderiv)


    return EER, logderiv, back





def EER2logdprime(EER,deriv = False):
    
    if not deriv: arg = Phi_inv(EER)
    else: arg, back1 = Phi_inv(EER,deriv=True)
    dprime = -2.0*arg
    logdprime = np.log(dprime)
    if not deriv: return logdprime
    def back(dlogprime): return back1(-2.0*dlogprime/dprime)
    return logdprime, back

def EER2dprime(EER,deriv = False):
    
    if not deriv: arg = Phi_inv(EER)
    else: arg, back1 = Phi_inv(EER,deriv=True)
    dprime = -2.0*arg
    if not deriv: return dprime
    def back(ddprime): return back1(-2.0*ddprime)
    return dprime, back






def reparam_pdf(x_pdf, y2x, y, cs = True):
    """
    Scalar PDF reparametrization.
    
    If y = f(x), where x ~ x_pdf
    then this function returns y_pdf(y).
    
    params:
        x_pdf: the probability density function for x
        y2x: the inverse function of f, equipped to compute derivatives
        y: a scalar or vector at which to evaluate y_pdf.
    
    """
    if np.iscomplexobj(y) or not cs:
        x, back = y2x(y, deriv=True)
        return x_pdf(x) * np.abs(back(1.0))
    else:
        cx = y2x(y + 1e-20j)
        x, dxdy = np.real(cx) , 1e20*np.imag(cx)
        return x_pdf(x) * np.abs(dxdy)
    


if __name__ == "__main__":
    print("Running test script for module erf_tools\n")

    from numpy.random import randn, rand
    from lib.deriv.adtools import cstest
    
    
    
    print("\nTesting erf:")
    x = randn(5)
    delta = cstest(erf,x)
    print(delta)
    
    print("\nTesting Phi:")
    x = randn(5)
    delta = cstest(Phi,x)
    print(delta)    
    
    
    print("\nTesting dprime2EER:")
    x = randn(5)
    delta = cstest(dprime2EER,x)
    print(delta)    
    
    print("\nTesting EER2dprime:")
    EER = rand(5)/2
    delta = cstest(EER2dprime,EER)
    print(delta)    
    EER2 = dprime2EER(EER2dprime(EER))
    print(abs(EER-EER2).max())




    print("\nTesting logdprime2EER:")
    x = randn(5)
    delta = cstest(logdprime2EER,x)
    print(delta)    
    

    print("\nTesting erfinv:")
    p = rand(5)
    delta = cstest(erfinv,p)
    print(delta)    
    
    print("\nTesting Phi_inv:")
    p = rand(5)
    delta = cstest(Phi_inv,p)
    print(delta)    
    
    print("\nTesting EER2logdprime:")
    p = rand(5)/2
    delta = cstest(EER2logdprime,p)
    print(delta)    



    print("\nTesting erf_and_logderiv:")
    x = randn(5)
    delta = cstest(erf_and_logderiv,x)
    print(delta)    

    print("\nTesting Phi_and_logderiv:")
    x = randn(5)
    delta = cstest(Phi_and_logderiv,x)
    print(delta)    
    
    print("\nTesting logdprime2EER_and_logabsderiv:")
    x = randn(5)
    delta = cstest(logdprime2EER_and_logabsderiv,x)
    print(delta)    
    EER1, logabsderiv = logdprime2EER_and_logabsderiv(x)
    EER2, back = logdprime2EER(x,deriv=True)
    print(abs(EER1-EER2).max())
    print(abs(logabsderiv-np.log(-back(1))).max())
    
    if False:

    
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        from scipy import stats
         
    #    y = np.linspace(-5,EER2logdprime(0.000001),200)
    #    pdf_x = stats.beta(1,1,scale=0.5).pdf      #uniform between 0 and 0.5
    #    y2x = logdprime2EER
    #    pdf_y = reparam_pdf(pdf_x,y2x,y)
    #    ax.plot(y,pdf_y,label="log(d')")
    
    #    y = np.linspace(np.exp(-5),np.exp(EER2logdprime(0.000001)),200)
    #    pdf_x = stats.beta(1,4,scale=0.5).pdf      #uniform between 0 and 0.5
    #    y2x = dprime2EER
    #    pdf_y = reparam_pdf(pdf_x,y2x,y)
    #    ax.plot(y,pdf_y,label="d'")
    
        y = np.linspace(0,0.49,200)
        pdf_x = stats.invgauss(mu = 1, scale = 1).pdf
        y2x = EER2dprime
        pdf_y = reparam_pdf(pdf_x,y2x,y)
        ax.plot(y,pdf_y,label="EER")
    
        ax.legend(loc='best', frameon=False)
        plt.xlabel("EER")
        plt.grid()
        plt.show() 
    

    
    
    
    