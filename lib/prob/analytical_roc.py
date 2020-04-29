"""
analytical_roc
"""

import numpy as np
import scipy.optimize as opt

from lib.special.erf_tools import dprime2EER, EER2dprime

def distributions2roc(tar,non,threshold):
        Pmiss = tar.cdf(threshold)
        Pfa = 1 - non.cdf(threshold)
        return Pmiss, Pfa
        
def distributions2EER(tar,non):
    
    def f(t):  # f is monotonic rising
        Pmiss, Pfa = distributions2roc(tar,non,t)
        return Pmiss - Pfa
    
    mutar = tar.stats("m").item()
    star = np.sqrt(tar.stats("v").item())
    munon = non.stats("m").item()
    snon = np.sqrt(non.stats("v").item())
    s = max(star,snon)
    
    top = mutar
    while not f(top) > 0: top = top + s
    
    bot = munon
    while not f(bot) < 0: bot = bot - s
    
    thr = opt.toms748(f, bot, top, disp = False)
    Pmiss, Pfa = distributions2roc(tar,non,thr)
    EER = (Pmiss + Pfa) /2
    
    return EER, thr 


def nominal_EER(tar,non):
    mutar = tar.stats("m").item()
    star = np.sqrt(tar.stats("v").item())
    munon = non.stats("m").item()
    snon = np.sqrt(non.stats("v").item())
    s = np.sqrt(star*snon)
    dprime = (mutar-munon) / s
    return dprime2EER(dprime)
    

    

if __name__ == "__main__":
    print("Running test script for module analytical_roc\n")
    
    from lib.prob import nig
    
#    gamma = 0.01
#    skew = 0
#    d_eer = 5 / 100
#    scalefac = 1.1
#
#    v = 1
#    munon = 0
#
#    dprime = EER2dprime(d_eer)
#    b = gamma * skew
#    v0 = (skew**2 + 1) / gamma
#    scale = np.sqrt(v/v0)
#    mutar = munon + np.sqrt(v) * dprime
#
#    scalenon = scale / scalefac 
#    scaletar = scale * scalefac
#    
#    locnon = munon - scalenon*skew
#    loctar = mutar - scaletar*skew


    L = 1
    R = 20
    d_eer = 5 / 100
    scalefac = 1.1

    v = 1
    munon = 0
    a = (L+R)/2
    b = (L-R) /2
    gamma = np.sqrt(a**2 - b**2)
    skew = b / gamma

    dprime = EER2dprime(d_eer)
    v0 = (skew**2 + 1) / gamma
    scale = np.sqrt(v/v0)
    mutar = munon + np.sqrt(v) * dprime

    scalenon = scale / scalefac 
    scaletar = scale * scalefac
    
    locnon = munon - scalenon*skew
    loctar = mutar - scaletar*skew


    
    non = nig.to_scipy_distr(np.log(scalenon),locnon,b,gamma)
    tar = nig.to_scipy_distr(np.log(scaletar),loctar,b,gamma)
    
    print("design EER : ", d_eer)
    print("nominal EER: ", nominal_EER(tar,non))
    eer, thr = distributions2EER(tar,non)
    print("actual EER : ",eer )
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)

    y = np.linspace(munon-4*scalenon,mutar+4*scaletar,200)
    ax[0].plot(y,non.pdf(y),label="non")
    ax[0].plot(y,tar.pdf(y),label="tar")
    ax[0].legend(loc='best', frameon=False)
    ax[0].grid()
    ax[1].plot(y,non.logpdf(y),label="non")
    ax[1].plot(y,tar.logpdf(y),label="tar")
    ax[1].grid()

    #plt.xlabel("EER = ", eer)
    plt.show() 
    


