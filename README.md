# netam-experiments-1

Experiments with [netam](https://github.com/matsengrp/netam)


## TODOs

* https://github.com/matsengrp/epam/issues/39
* compare models using the shmple framework
* make a per-NT model
* does performance depend on N padding?
* Consider the role of branch length. 
    * It's a normalization applied in training and evaluation. Prediction happens with branch length 1. 
    * We could have a better fitting model if we allowed branch length to vary.
* Think about boundary cases of beginning and end of sequence
    * you know, for BCR sequences, we could probably guess what the beginning and end is
* read up if others have done the same thing https://www.nature.com/articles/s41467-019-09027-x
    * DNABERT uses overlapping 3-mers https://doi.org/10.1093/bioinformatics/btab083
* train a mouse model
    * compare models predicting on the replay experiment?
* does the close mutations analysis add anything here?
* write an R interface for people who want to use the model
* set up CI


## Goals

* modern techniques such as regularization and transformers
* a comprehensive survey, and provide useful software with rigorous metrics
* comparison to oracle


## Results

* `cnn.ipynb`: Hyperparameter optimization for CNN model
* `cnn_1mer.ipynb`: Hyperparameter optimization for CNN1Mer model
* `cnnmlp.ipynb`: Adding a hidden layer in the final layer is bad
* `cnnpp.ipynb`: Adding a positional encoding to the CNN
* `cnnxformer.ipynb`: Adding a transformer to the CNN makes it worse
* `data-description.ipynb`: Exploration of SHMoof data sets
* `fivemer.ipynb`: L2 regularizing the 5mer model doesn't help
* `model_comparison.ipynb`: Main model comparison notebook
* `noof.ipynb`: A transformer on the kmer embeddings is not a good model
* `penalize-site-rates.ipynb`: Trying to penalize the site rates of SHMoof
* `persite_wrapper.ipynb`: Developing the `PersiteWrapper` and showing that regularizing it doesn't help
* `reshmoof.ipynb`: Re-fitting the SHMoof model, playing with regularization, showing that per-site mutability tracks per-site motif mutability
* `twolength.ipynb`: An experiment trying to see if stratifying the SHMoof model into long and short components doesn't help
 

## Conclusions
* CNN using kmer embeddings work, and can be parameter-sparse
* transformers aren't good for this problem, and positional encoding appears to hurt
* this doesn't really give credence to the idea that position along the sequence matters