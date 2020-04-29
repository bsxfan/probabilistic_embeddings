import numpy as np


class SingletonDict(dict):
    def __getitem__(self,key):
        return super().__getitem__(key) if key in self else {key}


class AHC:
    def __init__(self, X, B, w, prior):
        assert X.shape == B.shape
        n,d = X.shape
        
        self.n = self.N = n
        self.R = R = (w*B) / (w+B)
        self.RX = RX = R*X         #(n,d)
        
        self.LLH = (RX**2/(1.0+R) - np.log1p(R) ).sum(axis=1) / 2.0  #(n,)
        self.LLRs = []
        
        labels = np.arange(n,dtype=int) # full length labels, contains result
        self.ind = labels.copy() # 
        
        self.prior_ahc = prior.ahc(labels)
        
        # map every element to a singleton cluster containing that element
        self.clusters = SingletonDict()     
        
    def join(self,i,j):
        clusters = self.clusters
        join = clusters[i] | clusters[j]
        for e in join: clusters[e] = join
        
        
    def iteration(self, thr = 0.0):
        RX, R, n = self.RX, self.R, self.n
        prior_ahc, LLH = self.prior_ahc, self.LLH
        ind = self.ind
        
        #M = np.full((n,n),-np.Inf)
        
        maxval = -np.Inf
        for i in range(n-1):
            r = R[i,:]                    # (d,)      
            rR = r + R[i+1:,:]            # (n-i-1, d)
            rx = RX[i,:]
            rxRX = rx + RX[i+1:,:]
            llh = (rxRX**2/(1.0+rR) - np.log1p(rR) ).sum(axis=1) / 2.0  
            score = llh + prior_ahc.llr_joins(i) - LLH[i] - LLH[i+1:]
            #M[i,i+1:] = score
            j = score.argmax()
            scj = score[j]
            #print(i,i+j+1,': ',np.around(np.exp(scj),1))
            if scj > maxval:
                maxi = i
                maxj = j + i + 1
                maxval = scj
        
        #print(np.around(np.exp(M),1),'\n')
        LLRs = self.LLRs
        LLRs.append(maxval)         
          
        if maxval > thr:
            
            #print('joining: ',maxi,'+',maxj)
            #print('ind = ',ind)
            ii, jj = ind[maxi], ind[maxj]
            print('joining: ',ii,'+',jj)
            self.join(ii,jj)
            
            
            RX[maxi,:] += RX[maxj,:]
            R[maxi,:] += R[maxj,:]        
            self.RX = np.delete(RX,maxj,axis=0)         
            self.R = np.delete(R,maxj,axis=0)

            self.n = n-1

            prior_ahc.join(maxi,maxj) 

            LLH[maxi] = maxval + LLH[maxi] + LLH[maxj]
            self.LLH = np.delete(LLH,maxj)

            self.ind = np.delete(ind,maxj)
        
        return maxval
    
    
    def cluster(self, thr = 0.0):
        while self.n > 1:
            llr = self.iteration(thr)
            if llr <= thr: break
        #return clusters2labels(self.clusters,self.N)
        return self.labelclusters()
    
    
    def labelclusters(self):
        clusters, n = self.clusters, self.N
        labels = np.full(n,-1)
        label = -1
        for i in range(n):
            s = clusters[i]
            for e in s: break #get first set element
            if labels[e] < 0: 
                label += 1
                labels[list(s)] = label
        return labels    
    
    
    
    
    
    