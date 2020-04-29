import numpy as np

class PartitionsOf:
    """Iterator for all partitions of a set of n elements. Keep n small, 
       because the number of items is the Bell number, Bn.
    
    Usage: for labels in PartitionsOf(n):
               dosomething(labels)
               
    n: size of set to be partitioned            
               
               
    labels: an n-vector, with entries in {0,...,n-1}, such that each entry 
            identifies a block (or cluster, or subset) of the partition.
            It is a limited growth string: the first entry is zero. Every
            other entry can be at most 1 more than any of its predecessors.
            For fixed n, there is a 1-1 relationship between limited growth 
            strings and partitions of an n-set. 
            
    The first partition returned is the coarsest partition (having one block),
    represented by labels = np.zeros(n,int). The last partition is the finest 
    (having n blocks), represented by labels = np.arange(n,dtype=int).
                   
    
    """
    def __init__(self,n):
        self.n = n
        self.count = 0  # number of partitions visited so far
        self.labels = np.zeros(n,int)   # initialize first (coarsest) partition
        self.maxima = np.zeros(n,int)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        n, count, labels, maxima = self.n, self.count, self.labels, self.maxima
        if not count:           # first partition
            self.count = 1
            return labels
        
        # find next one
        for i in range(n-1,0,-1):
            if labels[i] <= maxima[i-1]:
                labels[i] += 1
                maxima[i] = max(maxima[i],labels[i])
                labels[i+1:] = 0
                maxima[i+1:] = maxima[i]
                self.count += 1
                return labels
            
        raise StopIteration
        

    def __repr__(self):
        return f"PartitionsOf({self.n})"
    

def PartitionsAsBlocklabels(n):
    """Returns iterator for all partitions of a set of n elements. Keep n small, 
       because the number of items is the Bell number, Bn.
    
    
    Usage: for blocks in PartitionsAsBlocklabels(n):
               dosomething(blocks)
               
    n: size of set to be partitioned            
    
    blocks: m-vector, with elements in {1,...,2**n-1}, where m is the number 
            of blocks in the current partition. Note: m varies between 1 and n. 
            Every element specifies a block in the partition. Block i has 
            element j if blocks[i] has bit j set.  


    """
    weights = 2**np.arange(n)
    f = lambda labels: labels_to_1hotmatrix(labels) @ weights
    return map(f,PartitionsOf(n))
    


    
def labels_to_1hotmatrix(labels, dtype=int):
    """
    Maps restricted growth string to a one-hot flag matrix. The input and 
    the output are equivalent representations of a partition of a set of
    n elelements. 
    
    labels: restricted growth string: n-vector with entries in {0,...,n-1}. 
            The first entry is 0. Other entries cannot exceed any previous 
            entry by more than 1.
            
    dtype: optional, default=int. Element data type for returned matrix. bool
           or float can also be used.       
           
    Returns (m,n) matrix, with 0/1 entries, where m is the number of blocks in 
            the partition and n is the numer of elements in the partitioned set.
            Columns are one-hot. If return_matrix[i,j], then element j is in 
            block i. 
           
    """
    m = 1 + labels.max()
    B = np.arange(m).reshape(-1,1) == labels
    return B.astype(dtype,copy=False) 
#    if not dtype==bool: 
#        R = np.empty(B.shape,dtype)
#        R[:] = B
#        return R
    
    
def blocklabels_to_flagvector(labels,sz,dtype=int):
    """
    Maps partition represented as a vector of block labels to a 0/1 flag 
    vector. The input and the output are equivalent representations of a 
    partition of a set of n elelements. 
    
    labels: m-vector with entries in {0,...,2**n-1}, where m is the number of 
            blocks in the partition. If bit j is set in labels[i], then element
            j of the partitioned set is in block i of the partition.
            
    sz: must be 2**n, where n is the size of the partitioned set         
            
    dtype: optional, default=int. Element data type for returned vector. bool
           or float can also be used.       
           
    Returns (m,n) matrix, with 0/1 entries, where m is the number of blocks in 
            the partition and n is the numer of elements in the partitioned set.
            Colums are one hot. If return_matrix[i,j], then element j is in 
            block i. 
           
    """
    r = np.zeros(sz,dtype)
    r[labels] = 1
    return r
    
    
    
    
def int2bits(i,n):
    """
    Returns n-vector with 0/1 entries, representing the bits in i. The bit 
    order is from least to most significant. Ensure i < 2*n, otherwise one or 
    more most significant bits will be ignored, e.g. int2bits(4,2) -> [0,0].
    """
    return (i>>np.arange(n,dtype=int))%2    


def Bell(n,k=0): 
    """Bell(n) is the n'th Bell number, the number of ways to partition a
    set of n elements.
    """
    return 1 if n==0 else k*Bell(n-1,k) + Bell(n-1,k+1)

def partitions_and_subsets(n,dtype=int):
    """
    Returns P,S, matrices with 0/1 elements. 
    P: (Bn,2**n): block j is in partition i if P[i,j]
    S: (2**n,n): block i has element j if S[i,j]
    
    """
    sz = 2**n
    f = lambda labels: blocklabels_to_flagvector(labels,sz,dtype)
    P = np.array([row for row in map(f,PartitionsAsBlocklabels(n))])
    S = np.array([int2bits(s,n) for s in range(2**n)]).astype(dtype,copy=False)
    return P,S

    
def create_index(n):
    """
    Returns f: RGS(n) --> {0,...,Bn-1}, i.e. a function that maps every
    restricted growth string of length n, to a unique integer in the range
    0 to Bn-1, where Bn is the n-th Bell number.
    
    The order is the same as that returned by PartitionsOf(n). 
    
    example:
        f = create_index(3)
        [f(rgs) for rgs in PartitionsOf(3)] --> [0,1,2,3,4]
    """
    index = {tuple(labels):i for i,labels in enumerate(PartitionsOf(n))}
    return lambda labels: index[tuple(labels)]
        
        
        