# Probabilistic Embeddings
This repository will contain Python code associated with our paper:

Anna Silnova, Niko Brummer, Johan Rohdin, Themos Stafylakis and Lukas Burget, "Probabilistic embeddings for speaker diarization", The Speaker and Language Recognition Workshop, Tokio, 2020.

The tools here will not allow you to fully replicate the experiments descibed in the paper. Specifically, we do not include the probabilistic x-vector extractor. We do include:

- A reference implementation of our discriminative traning criterion. It can be used to train the extractor and the PLDA backend, or just the backend. The criterion is a special case of multiclass cross-entropy, where the classes are all 4140 ways to partition a set of 8 elements.

- A reference implementation of our PLDA scoring algorithm. Given an n-tuple of embeddings extracted from an n-tuple of speech segments, the scoring algorithm can compute likelihoods of the form P(speech segment n-tuple | clustering hypothesis) .

- A reference implementation of the Chinese restaurant process prior on partition (clustering) hypotheses.

## What is a probabilistic embedding?
A traditional embedding is a vector in R<sup>n</sup>, with n fixed, extracted from some complex input (for us a speech segment of variable duration). Such embeddings are much easier to model and process than the original inputs.

The general desiderata for embeddings used in classification, clustering, verification, or any other recognition of discrete classes is that they should be: 
- Far apart (for example in Euclidean distance) when the are extacted from examples of different classes, 
- and close when they are from the same class.  
This ideal situation breaks down however when inputs might be of poor and/or variable quality. No matter how good the embedding extractor, it will not be able to procude neatly separated clusters of embeddings, where the clusters correspond to the true classes.
In such cases, the embedding extractor form a bottleneck that discards some important information about the quality of the input. 

In probabilistic embeddings, we augment each embedding with a vector of _precisions_ (also in R<sup>n</sup>), which is extrated jointly with the embedding by a modified embedding extractor. The idea is that precisions are high for those components of the embedding that could be reliably extracted and low (or zero) for those components that cannot be reliably extracted.

By viewing a probabilistic embedding with its precisions as diagonal Gaussian distribution for the values that the embedding might have had if extracted from a high quality input, we have shown in the paper how to derive probabilistic ways to evaluate the likelihoods for clustering hypotheses. 



