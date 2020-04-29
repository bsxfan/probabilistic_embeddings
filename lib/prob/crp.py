import numpy as np
from scipy.special import gammaln, psi
from numpy.random import choice
from scipy.special import expit as sigmoid
from scipy.special import logit
from scipy.optimize import minimize, minimize_scalar

from lib.special.softplus import softplus, softplusinv
from lib.combin.partitions import partitions_and_subsets

class CRP:
    def __init__(self,alpha,beta):
        assert alpha >= 0 <= beta <= 1
        self.alpha = alpha
        self.beta = beta
        
    def sample(self,n):
        alpha, beta = self.alpha, self.beta
        labels = np.zeros(n,int)  # restricted growth string, labels start at 0   
        counts = np.zeros(n,int)  # table occupancies (up to n tables)
        p = np.empty(n)
        counts[0] = 1             # seat first customer at table 0
        nt = 1                    # number of occupied tables  
        for i in range(1,n):      # seat rest of customers
            # i is number of seated customers and index of to-be-seated customer
            pi = p[:nt+1]
            pi[:nt] = (counts[:nt] - beta) / (i + alpha) # occupied tables
            pi[nt] = (alpha + nt*beta ) / (i + alpha)    # new table
            t = choice(nt+1, None, True, pi)             # chosen table
            labels[i] = t
            counts[t] += 1
            if t == nt: nt += 1                          # new table was chosen
        return labels, counts[:nt]    
    
    def samples(self,n,m):
        """
        Sample m independent partitions of size n from this CRP
        Returns a list of m arrays of block sizes.
        (The array sizes are variable, depending on the number of blocks in 
        the partition.)
        """
        assert n >= 1 <= m
        counts_list = []
        for i in range(m):
            labels, counts = self.sample(n)
            counts_list.append(counts)
        return counts_list
    
    def logprob(self,counts):
        """
        Returns log P(labels | crp), where labels is represented by the given
        table occupancy counts.
        """
        alpha, beta = self.alpha, self.beta
        
        if alpha == np.Inf and beta==1: #singleton tables
            return 0.0 if all(counts==1) else -np.Inf
        
        if alpha==0 and beta==0:       #single table
            return 0.0 if len(counts)==1 else -np.Inf
        
        if alpha>0 and beta>0:  # general case (2 parameter Pitman-Yor CRP)
            return logprob_alpha_beta(alpha,beta,counts)
  
        if beta==0 and alpha>0:  # classical 1-parameter CRP
           return logprob_alpha(alpha,counts)
       
        if beta>0 and alpha==0:
            return logprob_beta(beta,counts)
        
        assert False
        
    def llr_joins(self,counts,i):
        """
        Let logLR(i,j) = log P(join(i,j)|crp) - log P( labels| crp), where
        labels is represented by the given occupancy counts; and where join(i,j)
        joins tables i and j, while leaving other tables as-is. A vector is 
        returned with all logLR(i,j), with j > i. 
        
        For use by AHC (agglomerative hierarchical clustering) algorithms that 
        seek greedy MAP partitions, where this CRP forms the partition prior.
        """
        alpha, beta = self.alpha, self.beta
        K = len(counts)  # tables
        assert K > 1
        ci = counts[i]
        cj = counts[i+1:]
        llr = gammaln(1-beta) - np.log(beta) - np.log(alpha/beta + K-1) 
        llr += gammaln(cj+(ci-beta)) - gammaln(ci-beta) - gammaln(cj-beta) 
        return llr
            

    
    def exp_num_tables(self,n):
        """
        n: number of customers
        """
        alpha, beta = self.alpha, self.beta
        if alpha==0 and beta==0:
            e = 1
        elif alpha == np.Inf:
            e = n
        elif alpha>0 and beta>0:      
            A = gammaln(alpha + beta + n) + gammaln(alpha + 1) \
                - np.log(beta) - gammaln(alpha+n) - gammaln(alpha+beta)
            B = alpha/beta
            e = B*np.expm1(A-np.log(B))   # exp(A)-B
        elif alpha>0 and beta==0:
            e = alpha*( psi(n+alpha) - psi(alpha) )
        elif alpha==0 and beta>0:
            A = gammaln(beta + n) - np.log(beta) - gammaln(n) - gammaln(beta)
            e = np.exp(A)
        return e
    
    def __repr__(self):
        return f"CRP(alpha={self.alpha}, beta={self.beta})"
    
    
    def logprobtable(self,P,S=None):
        """
        Returns pre-computed table of log-probabilities, for every partition
        of a set of n elements. 
        
        Usage:
            
            P, S = lib.combin.partitions_and_subsets(n,dtype=bool)
            table = crp.logprobtable(P,S)
        
        or, equivalently:
            
            table = crp.logprobtable(n)
        
        """
        if S is None: 
            assert type(P)==int and P>0
            P,S = partitions_and_subsets(P,dtype=bool)
        counts = S.sum(axis=1)
        Bn, ns = P.shape
        L = [self.logprob(counts[P[i,:].astype(bool,copy=False)]) for i in range(Bn)]
        return np.array(L)
    
    
    def ahc(self,labels):
        """
        Returns an AHC object, initialized at the given labels.
        
        For use by AHC (agglomerative hierarchical clustering) algorithms that 
        seek greedy MAP partitions, where this CRP forms the partition prior.
        
        """
        return AHC(self,labels)
    
    


class CRPalpha(CRP):
    """
    Classical 1-parameter CRP, with beta=0. Gives narrower distributions for
    numbers of tables.
    """
    def __init__(self,alpha): super().__init__(alpha,0)
    
    def logprob(self,counts):
        alpha = self.alpha
        if alpha==0:       #single table
            return 0.0 if len(counts)==1 else -np.Inf
        return logprob_alpha(alpha,counts)
    
    def llr_joins(self,counts,i):
        alpha = self.alpha
        K = len(counts)  # tables
        assert K > 1
        ci = counts[i]
        cj = counts[i+1:]
        llr = - np.log(alpha)
        llr += gammaln(cj+ci) - gammaln(ci) - gammaln(cj) 
        return llr

    
    
    
    def exp_num_tables(self,n):
        """
        n: number of customers
        """
        alpha = self.alpha
        if alpha==0: return 1
        if alpha == np.Inf: return n
        return alpha*( psi(n+alpha) - psi(alpha) )

class CRPbeta(CRP):
    """
    Special case of CRP, with alpha=0. Gives wider distributions for
    numbers of tables.
    """
    def __init__(self,beta): super().__init__(0,beta)
    
    def logprob(self,counts):
        beta = self.beta
        if beta==0:       #single table
            return 0.0 if len(counts)==1 else -np.Inf
        return logprob_beta(beta,counts)
    
    def exp_num_tables(self,n):
        """
        n: number of customers
        """
        beta = self.beta
        if beta==0: return 1
        A = gammaln(beta + n) - np.log(beta) - gammaln(n) - gammaln(beta)
        return np.exp(A)





    
    
class AHC:
    """
        For use by AHC (agglomerative hierarchical clustering) algorithms that 
        seek greedy MAP partitions, where this CRP forms the partition prior.
    """    
    def __init__(self,crp,labels):
        self.crp = crp
        tables, counts = np.unique(labels,return_counts=True)
        self.counts = counts
        
    def llr_joins(self,i):
        """
        Scores in logLR form, the CRP prior's contribution when joining tabls 
        i with all tables j > i.
        """
        crp, counts = self.crp, self.counts
        return crp.llr_joins(counts,i)
    
    def join(self,i,j):
        """
        Joins tables i and j in this AHC object.
        """
        counts = self.counts
        counts[i] += counts[j]
        self.counts = np.delete(counts,j)
        
    
        


def logprob_alpha_beta(alpha,beta,counts):
    if not 0 < beta < 1: print("beta = ",beta)
    assert alpha > 0 < beta < 1
    K, T = len(counts), sum(counts)  # tables, customers
    logP = gammaln(alpha) - gammaln(alpha+T) + K*np.log(beta) \
         + gammaln(alpha/beta + K) - gammaln(alpha/beta)  \
         + sum(gammaln(counts-beta)) \
         - K*gammaln(1-beta)
    return logP


def logprob_alpha(alpha,counts):
    """
    beta assumed = 0
    """
    assert alpha > 0
    K, T = len(counts), sum(counts)  # tables, customers
    logP = gammaln(alpha) + K*np.log(alpha) - gammaln(alpha+T) \
         + sum(gammaln(counts))
    return logP

def logprob_beta(beta,counts):
    """
    alpha assumed = 0
    """
    assert 0 < beta < 1
    K, T = len(counts), sum(counts)  # tables, customers
    logP = (K-1)*np.log(beta) + gammaln(K) - gammaln(T) \
         - K*gammaln(1-beta) + sum(gammaln(counts-beta))
    return logP


def logprob_obj(alpha_beta, counts_list):
    s = 0.0
    spi_alpha, logit_beta = alpha_beta
    alpha = softplus(spi_alpha) 
    beta = sigmoid(logit_beta)
    for counts in counts_list:
        s -= logprob_alpha_beta(alpha,beta,counts)
    return s    

def ML_crp(counts_list):
    params = np.zeros(2)
    print("ML optimization starting at obj = ",logprob_obj(params,counts_list))
    res = minimize(logprob_obj,params,args=(counts_list))
    params = res.x
    print("ML optimization ended at obj = ",logprob_obj(params,counts_list))
    spi_alpha, logit_beta = params
    alpha = softplus(spi_alpha) 
    beta = sigmoid(logit_beta)
    return CRP(alpha,beta)


def ML_beta_crp(counts_list):
    def obj(logit_beta):
        beta = sigmoid(logit_beta)
        s = 0.0
        for counts in counts_list:
            s -= logprob_beta(beta,counts)
        return s    
    print("ML optimization starting at obj = ",obj(0.0))
    res = minimize_scalar(obj)  # starts at logit_beta = 0
    beta = sigmoid(res.x)
    print("ML optimization ended at obj = ",obj(res.x))
    return CRP(0,beta)

def ML_alpha_crp(counts_list):
    def obj(spi_alpha):
        alpha = softplus(spi_alpha)
        s = 0.0
        for counts in counts_list:
            s -= logprob_alpha(alpha,counts)
        return s    
    res = minimize_scalar(obj)  # starts at logit_beta = 0
    alpha = softplus(res.x)
    return CRP(alpha,0)



if __name__ == "__main__":
    print("Running test script for module crp\n")
    
#    alpha = 30
#    beta = 0.8
#    crp0 = CRP(alpha,beta)
#    print("crp0: ",crp0)
#    print("sampling:")
#    samples = crp0.samples(100,100)
#    params = np.array([softplusinv(crp0.alpha),logit(crp0.beta)])
#    print("crp0 obj = ",logprob_obj(params,samples))    
#    crp1 = ML_crp(samples)
#    print("crp1: ",crp1)
#    
#    crp2 = ML_beta_crp(samples)
#    print("crp2: ",crp2)
    
    
    
    
    
    




    