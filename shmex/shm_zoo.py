import os
import sys

import numpy as np
import pandas as pd
import random
import torch

torch.set_num_threads(1)

from netam.common import pick_device
from netam.framework import (
    SHMoofDataset,
    RSSHMBurrito,
)
from netam import models

sys.path.append("..")
from shmex.shm_data import train_val_dfs_of_nicknames

# Very helpful for debugging!
# torch.autograd.set_detect_anomaly(True)

site_count = 500
epochs = 1000
device = pick_device()

model_parameters = {
    "sml": {
        "kmer_length": 3,
        "kernel_size": 7,
        "embedding_dim": 6,
        "filter_count": 14,
        "dropout_prob": 0.1,
    },
    "med": {
        "kmer_length": 3,
        "kernel_size": 9,
        "embedding_dim": 7,
        "filter_count": 16,
        "dropout_prob": 0.2,
    },
    "lrg": {
        "kmer_length": 3,
        "kernel_size": 11,
        "embedding_dim": 7,
        "filter_count": 19,
        "dropout_prob": 0.3,
    },
    "4k": {
        "kmer_length": 3,
        "kernel_size": 17,
        "embedding_dim": 12,
        "filter_count": 14,
        "dropout_prob": 0.1,
    },
    "4k13": {
        "kmer_length": 3,
        "kernel_size": 13,
        "embedding_dim": 12,
        "filter_count": 20,
        "dropout_prob": 0.3,
    },
    "8k": {
        "kmer_length": 3,
        "kernel_size": 15,
        "embedding_dim": 14,
        "filter_count": 25,
        "dropout_prob": 0.0,
    },
}

kmer_size_from_model_type = {
    "cnn": 3,
    "fivemer": 5,
}

default_burrito_params = {
    "batch_size": 1024,
    "learning_rate": 0.001,
    "min_learning_rate": 1e-6,  # early stopping!
    "weight_decay": 1e-6,
}


def kmer_size_from_model_name(model_name):
    return kmer_size_from_model_type[model_name.split("_")[-1]]


def create_model(model_name):
    if model_name == "fivemer":
        model = models.RSFivemerModel()
    elif model_name == "rsshmoof":
        model = models.RSSHMoofModel(kmer_length=5, site_count=site_count)
    elif model_name.startswith("cnn_"):
        model_name = model_name[4:]
        hparam_name = model_name[4:]
        if hparam_name not in model_parameters:
            raise ValueError(f"Unknown hparam key: {hparam_name}")
        if model_name.startswith("ind_"):
            model = models.IndepRSCNNModel(**model_parameters[hparam_name])
        elif model_name.startswith("hyb_"):
            model = models.HybridRSCNNModel(**model_parameters[hparam_name])
        elif model_name.startswith("joi_"):
            model = models.JoinedRSCNNModel(**model_parameters[hparam_name])
        else:
            raise ValueError(f"Unknown model key: {model_name}")
    else:
        raise ValueError(f"Unknown model key: {model_name}")
    model.to(device)
    return model


def trained_model_str(model_name, data_nickname, training_method, seed):
    return f"{model_name}-{data_nickname}-{training_method}-{seed}"


def trained_model_path(model_name, data_nickname, training_method, seed):
    return f"trained_models/{trained_model_str(model_name, data_nickname, training_method, seed)}"


def fixed_model_path(model_name):
    return f"fixed_models/{model_name}"


def train_model(
    model_name,
    dataset_name,
    training_method,
    seed,
    crepe_dest_path=None,
    val_is_train=False,
    **burrito_kwargs,
):
    """
    Our goal with the seed is to ensure the different trainings are independent,
    not to ensure reproducibility. See comments below.

    val_is_train (bool, optional): If True, then we use all the data for training and make the validation set is the same as the training set. We only do this when we are training a final model and want to use all of the data. Defaults to False.
    """
    # Update default parameters with any provided keyword arguments
    burrito_params = {**default_burrito_params, **burrito_kwargs}

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # If we want to ensure reproducibility, we would also set the following:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    train_df, val_df = train_val_dfs_of_nicknames(
        dataset_name
    )
    if val_is_train:
        train_df = pd.concat([train_df, val_df])
        val_df = train_df.copy()
    if crepe_dest_path is None:
        crepe_dest_path = trained_model_path(
            model_name, dataset_name, training_method, seed
        )
    if "/" in crepe_dest_path:
        crepe_dest_dir = crepe_dest_path[: crepe_dest_path.rfind("/")]
        assert os.path.exists(
            crepe_dest_dir
        ), f"Directory `{crepe_dest_dir}` does not exist"

    model = create_model(model_name)
    burrito = RSSHMBurrito(
        SHMoofDataset(train_df, kmer_length=model.kmer_length, site_count=site_count),
        SHMoofDataset(val_df, kmer_length=model.kmer_length, site_count=site_count),
        model,
        **burrito_params,
        name=trained_model_str(model_name, dataset_name, training_method, seed),
    )
    burrito.train_dataset.to(device)
    burrito.val_dataset.to(device)

    if dataset_name.startswith("tst"):
        burrito.joint_train(epochs=2, training_method=training_method)
    elif training_method == "simple":
        burrito.simple_train(epochs=epochs)
    else:
        burrito.joint_train(
            epochs=epochs, training_method=training_method, cycle_count=5
        )

    burrito.save_crepe(crepe_dest_path)
    burrito.train_dataset.export_branch_lengths(
        crepe_dest_path + ".train_branch_lengths.csv"
    )
    burrito.val_dataset.export_branch_lengths(
        crepe_dest_path + ".val_branch_lengths.csv"
    )

    return model


def standardize_and_optimize_branch_lengths(model, pcp_df):
    """
    Given a model and a DataFrame of parent-child pairs, standardize the rates
    in the model and then optimize the branch lengths, updating the column in
    the pcp_df.

    This is used for model evaluation, so we make the optimization tolerance
    smaller.
    """
    burrito = RSSHMBurrito(
        None,
        SHMoofDataset(pcp_df, kmer_length=model.kmer_length, site_count=site_count),
        model,
    )
    burrito.standardize_and_optimize_branch_lengths(optimization_tol=1e-4)

    pcp_df["orig_branch_length"] = pcp_df["branch_length"]
    pcp_df["branch_length"] = burrito.val_dataset.branch_lengths
    return pcp_df


# Dictionaries for translation
model_translations = {
    "fivemer": "5mer",
    "rsshmoof": "Spisak",
    "cnn": "CNN",
    "ind": "Indep",
    "joi": "Joined",
    "hyb": "Hybrid",
    "sml": "Small",
    "med": "Medium",
    "lrg": "Large",
    "4k": "4K",
}


def long_name_of_short_name(short_name):
    parts = short_name.split("_")
    # Translate each part using the model_translations dictionary
    full_name_parts = [model_translations.get(part, part) for part in parts]
    # Special handling for models without underscores
    if len(full_name_parts) == 1:
        full_name = model_translations.get(short_name, short_name)
    else:
        # Join the translated parts with spaces for CNN models
        full_name = " ".join(full_name_parts[:-1]) + " " + full_name_parts[-1]

    return full_name


def fix_parameter_count(row):
    if row["model"] in ["rsshmoof", "origshmoof"]:
        # For every row with rsshmoof as the model subtract off 4**5 from the
        # parameter count, correspding to every possible 5mer getting mutated to
        # itself. We handle this setting indirectly by zeroing out the WT
        # prediction.
        return row["parameter_count"] - 4**5
    elif row["model"] in ["fivemer"]:
        # This corresponds to every 5mer being mutated to itself, and also
        # the fact that for the parameterization we're using rates and
        # conditional probabilities, which are only used as a product
        # so the effective number of parameters is less.
        return row["parameter_count"] - 2 * 4**5
    else:
        return row["parameter_count"]
