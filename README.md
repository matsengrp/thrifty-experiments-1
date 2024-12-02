# thrifty-experiments-1

SHM experiments with the [netam](https://github.com/matsengrp/netam) package as described in the manuscript [_Thrifty wide-context models of B cell receptor somatic hypermutation_](https://www.biorxiv.org/content/10.1101/2024.11.26.625407v1) by Sung, et al.

Start by creating an environment and installing netam into it as described in the `netam` README.

To do the analysis described here, you will need to install the `shmex` package contained in this repository. 
To do so, clone this repository and install the package with from the root of this repository.

    pip install -e .

In an environment into which you have installed `epam` and `netam`.


## Local configuration and data

First download data from Dryad [here](https://doi.org/10.5061/dryad.np5hqc044).

Then edit `shmex/local.py` to reflect your directory structure.

To run a small trial analysis to see if things are working, enter the `train` directory and execute

    snakemake -cN --configfile config_test.yml


## Running the primary experiments

To train the main models and do the validation, enter the `train` directory and execute

    snakemake -cN

This will run the analysis on `N` cores (substitute your desired number of cores for `N`).

To run the more limited analysis on all the models, enter the `train` directory and execute

    snakemake -cN --configfile config_human_all.yml


## Notebook-based experiments

Other associated experiments are in the following notebooks. 

* `cnnpe.ipynb`: Adding a positional encoding to the CNN
* `cnnxformer.ipynb`: Adding a transformer to the CNN makes it worse
* `crepe_of_shmoof.ipynb`: Fitting the original Spisak et al. model weights into the framework used here
* `data-description.ipynb`: Exploration of SHMoof data sets
* `model_summaries.ipynb`: Summarizing model shapes
* `multihit_*`: Multihit analysis to be described in a future manuscript
* `neutral_codon.ipynb`: Also part of the multihit analysis
* `noof.ipynb`: A transformer on the kmer embeddings is not a good model
* `performance.ipynb`: Main model comparison notebook
* `persite_wrapper.ipynb`: Developing the `PersiteWrapper` that adds a per-site component to a model and showing that regularizing it doesn't help
* `reshmoof.ipynb`: Re-fitting the SHMoof model, playing with regularization, showing that per-site mutability tracks per-site motif mutability
* `shm_oe.ipynb`: Oberved/expected plotting
