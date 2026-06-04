# Experiments Repository for paper: "Thrifty Wide-Context Models of B Cell Receptor Somatic Hypermutation"

This repository contains the code and data for reproducing the experiments from:

**"Thrifty wide-context models of B cell receptor somatic hypermutation"**
*Published in eLife*
DOI: [10.7554/eLife.105471](https://doi.org/10.7554/eLife.105471)

Authors: Kevin Sung, Mackenzie M. Johnson, Will Dumm, Noah Simon, Hugh K. Haddox, Julia Fukuyama, Frederick A. Matsen IV

## Overview

This archive contains all the necessary notebooks, trained models, and supporting code to reproduce the figures and analyses from the paper. The models predict somatic hypermutation patterns in B cell receptors using efficient wide-context architectures.

## Installation

The setup below has been run with Python 3.12.

### 1. Install this package

```bash
git clone https://github.com/matsengrp/thrifty-experiments-1
cd thrifty-experiments-1
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .
cd -
```

### 2. Install epam

```bash
git clone https://github.com/matsengrp/epam
cd epam
make install
cd -
```

### 3. Install netam

```bash
git clone https://github.com/matsengrp/netam
cd netam
git checkout 030ceee
make install
cd -
```

### 4. Configure local paths and data

Download data from Dryad: <https://doi.org/10.5061/dryad.np5hqc044>

Edit `LOCAL_PATHS` dictionary in `thrifty-experiments-1/shmex/local.py` to set paths appropriate for your system. Specifically,
- edit `DATA_DIR` to the path containing the contents of the Dryad archive
- edit `FIGURES_DIR` to where some notebooks will write some outputs to


## Repository Structure

```
├── data/                    # S5F model files and a small simulation dataset
├── deepshm/                 # Data and scripts for comparison with DeepSHM
├── human/                   # Analysis notebooks on human data
├── shmex/                   # Python package with analysis utilities
├── train/                   # Model training pipeline and configurations
└── README.md                # This file
```

## Running the primary experiments

The model training pipelines are to be run in `train/`.

### Quick test

To run a small trial analysis to verify your setup:

```bash
snakemake -cN --configfile config_test.yml  # Replace N with desired number of cores
```

### Main model training and validation

```bash
snakemake -cN  # Replace N with desired number of cores
```

### Extended analysis on all models

```bash
snakemake -cN --configfile config_human_all.yml  # Replace N with desired number of cores
```

## Key Notebooks

The main analyses from the paper can be reproduced using these notebooks in `human/`:

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
