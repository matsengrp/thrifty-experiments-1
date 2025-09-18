# Experiments Repository for paper: "Thrifty Wide-Context Models of B Cell Receptor Somatic Hypermutation"

This repository contains the code and data for reproducing the experiments from:

**"Thrifty wide-context models of B cell receptor somatic hypermutation"**
*Published in eLife*
DOI: [10.7554/eLife.105471](https://doi.org/10.7554/eLife.105471)

Authors: Kevin Sung, Mackenzie M. Johnson, Will Dumm, Noah Simon, Hugh K. Haddox, Julia Fukuyama, Frederick A. Matsen IV

## Overview

This archive contains all the necessary notebooks, trained models, and supporting code to reproduce the figures and analyses from the paper. The models predict somatic hypermutation patterns in B cell receptors using efficient wide-context architectures.

## Installation

### 1. Install netam

First, install the [netam](https://github.com/matsengrp/netam) package in a virtual environment as described in the netam README.

### 2. Install this package

```bash
# Clone this repository
git clone this repository
cd thrifty-experiments-1

# Install in development mode
pip install -e .
```

Note: Ensure you have both `epam` and `netam` installed in your environment.

### 3. Configure local paths and data

Download data from Dryad: <https://doi.org/10.5061/dryad.np5hqc044>

Edit `shmex/local.py` to set paths appropriate for your system.

## Running the primary experiments

### Quick test

To run a small trial analysis to verify your setup:

```bash
cd train
snakemake -cN --configfile config_test.yml
```

### Main model training and validation

```bash
cd train
snakemake -cN  # Replace N with desired number of cores
```

### Extended analysis on all models

```bash
cd train
snakemake -cN --configfile config_human_all.yml
```

## Key Notebooks

The main analyses from the paper can be reproduced using these notebooks:

- **`performance.ipynb`** - Main model comparison notebook
- **`shm_oe.ipynb`** - Observed/expected plotting
- **`data-description.ipynb`** - Exploration of SHMoof data sets
- **`model_summaries.ipynb`** - Summarizing model shapes
- **`reshmoof.ipynb`** - Re-fitting the SHMoof model, playing with regularization, showing that per-site mutability tracks per-site motif mutability

### Additional experiments

- **`cnnpe.ipynb`** - Adding a positional encoding to the CNN
- **`cnnxformer.ipynb`** - Adding a transformer to the CNN makes it worse
- **`crepe_of_shmoof.ipynb`** - Fitting the original Spisak et al. model weights into the framework used here
- **`noof.ipynb`** - A transformer on the kmer embeddings is not a good model
- **`persite_wrapper.ipynb`** - Developing the `PersiteWrapper` that adds a per-site component to a model and showing that regularizing it doesn't help
- **`multihit_*`** - Multihit analysis to be described in a future manuscript
- **`neutral_codon.ipynb`** - Also part of the multihit analysis

## Repository Structure

```
├── train/                   # Model training pipeline and configurations
├── shmex/                   # Python package with analysis utilities
├── notebooks/               # Analysis notebooks
└── README.md               # This file
```

## Citation

If you use this code or data, please cite:

```bibtex
@article{Sung2025-hz,
  title={Thrifty wide-context models of {B} cell receptor somatic hypermutation},
  author={Sung, Kevin and Johnson, Mackenzie M and Dumm, Will and Simon, Noah and Haddox, Hugh K and Fukuyama, Julia and Matsen IV, Frederick A},
  journal={eLife},
  volume={14},
  pages={RP105471},
  month={aug},
  year={2025},
  doi={10.7554/eLife.105471},
  pmid={40879331}
}
```
