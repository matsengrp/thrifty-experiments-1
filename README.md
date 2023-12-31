# netam-experiments-1

Experiments with [netam](https://github.com/matsengrp/netam)


## TODOs

* move only a batch at a time to GPU
* we assume that if a sequence is N in the child it is also N in the parent

* compare models using the shmple framework
* make a per-NT model
* does performance depend on N padding?
* For branch length, the best approach is to optimize and normalize a given site of a given sequence to 1
* Think about boundary cases of beginning and end of sequence
    * you know, for BCR sequences, we could probably guess what the beginning and end is
* read up if others have done the same thing https://www.nature.com/articles/s41467-019-09027-x
    * DNABERT uses overlapping 3-mers https://doi.org/10.1093/bioinformatics/btab083
    * MuRaL uses k-mer embedding, and a CNN in parallel, but not in series https://www.nature.com/articles/s42256-022-00574-5
* does the close mutations analysis add anything here?
* write an R interface for people who want to use the model
* set up CI

* Future work could use structural information for the dnsm
    * https://www.mlsb.io/papers_2023/Enhancing_Antibody_Language_Models_with_Structural_Information.pdf

### DNSM

* refactor so we can compare selection models between flairr and 10x


## Goals

* modern techniques such as regularization and transformers
* a comprehensive survey, and provide useful software with rigorous metrics
* comparison to oracle


## Results

* `cnn_1mer.ipynb`: Hyperparameter optimization for CNN1Mer model
* `cnnmlp.ipynb`: Adding a hidden layer in the final layer is bad
* `cnnpp.ipynb`: Adding a positional encoding to the CNN
* `cnnxformer.ipynb`: Adding a transformer to the CNN makes it worse
* `data-description.ipynb`: Exploration of SHMoof data sets
* `fivemer.ipynb`: L2 regularizing the 5mer model doesn't help
* `hyper.ipynb`: Hyperparameter optimization for CNN model
* `model_comparison.ipynb`: Main model comparison notebook
* `noof.ipynb`: A transformer on the kmer embeddings is not a good model
* `penalize-site-rates.ipynb`: Trying to penalize the site rates of SHMoof
* `persite_wrapper.ipynb`: Developing the `PersiteWrapper` and showing that regularizing it doesn't help
* `reshmoof.ipynb`: Re-fitting the SHMoof model, playing with regularization, showing that per-site mutability tracks per-site motif mutability
* `twolength.ipynb`: An experiment trying to see if stratifying the SHMoof model into long and short components would help, but it doesn't


## Conclusions
* CNN using kmer embeddings work, and can be parameter-sparse
* transformers aren't good for this problem, and positional encoding appears to hurt
* this doesn't really give credence to the idea that position along the sequence matters
