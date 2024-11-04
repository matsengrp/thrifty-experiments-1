# thrifty-experiments-1

SHM experiments with the [netam](https://github.com/matsengrp/netam) package as described in the manuscript ``Thrifty wide-context models of B cell receptor somatic hypermutation'' by Sung, et al.

Start by creating an environment and installing netam into it as described in the `netam` README.

To do the analysis described here, you will need to install the `shmex` package contained in this repository. 
To do so, clone this repository and install the package with from the root of this repository.

    pip install -e .

In an environment into which you have installed `epam` and `netam`.


## Local configuration and data

First download data using the Dryad link associated with the paper.

Then edit `shmex/local.py` to reflect your directory structure.

To run a small trial analysis to see if things are working, enter the `train` directory and execute

    snakemake -cN --configfile config_test.yml


## Running the primary experiments

To train the main models and do the validation, enter the `train` directory and execute

    snakemake -cN

This will run the analysis on `N` cores (substitute your desired number of cores for `N`).

To run the more limited analysis on all the models, enter the `train` directory and execute

    snakemake -cN --configfile config_human_all.yml


## Results

Other associated experiments are in the following notebooks. 

* `cnnpe.ipynb`: Adding a positional encoding to the CNN
* `cnnxformer.ipynb`: Adding a transformer to the CNN makes it worse
* `data-description.ipynb`: Exploration of SHMoof data sets
* `noof.ipynb`: A transformer on the kmer embeddings is not a good model
* `performance.ipynb`: Main model comparison notebook
* `persite_wrapper.ipynb`: Developing the `PersiteWrapper` that adds a per-site component to a model and showing that regularizing it doesn't help
* `reshmoof.ipynb`: Re-fitting the SHMoof model, playing with regularization, showing that per-site mutability tracks per-site motif mutability
crepe_of_shmoof.ipynb
model_summaries.ipynb
multihit_extensions.ipynb multihit_model_exploration.ipynb multihit_use.ipynb
neutral_codon.ipynb
shm_oe.ipynb
