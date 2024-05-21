# netam-experiments-1

Experiments with [netam](https://github.com/matsengrp/netam)

To install

    pip install -e .

In an environment into which you have installed `epam` and `netam`.


## Goals

* modern techniques such as regularization and transformers
* a comprehensive survey, and provide useful software with rigorous metrics
* comparison to oracle


## Results

* `cnn_1mer.ipynb`: Hyperparameter optimization for CNN1Mer model
* `cnnmlp.ipynb`: Adding a hidden layer in the final layer is bad
* `cnnpe.ipynb`: Adding a positional encoding to the CNN
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
