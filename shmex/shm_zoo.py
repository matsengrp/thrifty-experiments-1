import os
import sys

import torch

torch.set_num_threads(1)

from netam.common import pick_device
from netam.framework import (
    SHMoofDataset,
    RSSHMBurrito,
)
from netam import models

sys.path.append("..")
from shmex.shm_data import train_val_dfs_of_nickname

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
    "4k_k13": {
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
    return model


burrito_params = {
    "batch_size": 1024,
    "learning_rate": 0.1,
    "min_learning_rate": 1e-6,  # early stopping!
    "l2_regularization_coeff": 1e-6,
}


def trained_model_str(model_name, data_nickname, training_method):
    return f"{model_name}-{data_nickname}-{training_method}"


def trained_model_path(model_name, data_nickname, training_method):
    return f"trained_models/{trained_model_str(model_name, data_nickname, training_method)}"


def train_model(model_name, dataset_name, training_method, crepe_dest_path=None):
    train_df, val_df = train_val_dfs_of_nickname(dataset_name)
    if crepe_dest_path is None:
        crepe_dest_path = trained_model_path(model_name, dataset_name, training_method)
    if "/" in crepe_dest_path:
        crepe_dest_dir = crepe_dest_path[: crepe_dest_path.rfind("/")]
        assert os.path.exists(crepe_dest_dir), f"Directory `{crepe_dest_dir}` does not exist"

    model = create_model(model_name)
    burrito = RSSHMBurrito(
        SHMoofDataset(train_df, kmer_length=model.kmer_length, site_count=site_count),
        SHMoofDataset(val_df, kmer_length=model.kmer_length, site_count=site_count),
        model,
        **burrito_params,
        name=trained_model_str(model_name, dataset_name, training_method),
    )

    if dataset_name == "tst":
        burrito.joint_train(epochs=2, training_method=training_method)
    else:
        burrito.joint_train(epochs=epochs, training_method=training_method)

    burrito.save_crepe(crepe_dest_path)
    burrito.train_loader.dataset.export_branch_lengths(crepe_dest_path + ".train_branch_lengths.csv")
    burrito.val_loader.dataset.export_branch_lengths(crepe_dest_path + ".val_branch_lengths.csv")

    return model
