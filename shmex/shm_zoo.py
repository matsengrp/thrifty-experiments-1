import os
import sys

import pandas as pd

from netam.common import pick_device
from netam.framework import load_crepe, SHMoofDataset, RSSHMBurrito
from netam import models 

import torch
torch.set_num_threads(1)

sys.path.append("..")
from shmex.shm_data import load_shmoof_dataframes, pcp_df_of_nickname

# Very helpful for debugging!
# torch.autograd.set_detect_anomaly(True)

shmoof_path = "~/data/shmoof_pcp_2023-11-30_MASKED.csv"
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
    }
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


def trained_model_str(model_name, data_nickname):
    return f"{model_name}-{data_nickname}"


def trained_model_path(model_name, data_nickname):
    return f"trained_models/{trained_model_str(model_name, data_nickname)}"


def train_model(model_name, dataset_name, resume=True):
    if dataset_name == "tst":
        sample_count = 1000
        val_nickname = "small"
    else:
        sample_count = None
        shmoof, val_nickname = dataset_name.split("_")
        assert shmoof == "shmoof"
    train_df, val_df = load_shmoof_dataframes(shmoof_path, sample_count=sample_count, val_nickname=val_nickname)
    out_prefix = trained_model_path(model_name, dataset_name)
    # burrito_name = trained_model_str(model_name, dataset_name)
    model = create_model(model_name)
    burrito = RSSHMBurrito(
        SHMoofDataset(train_df, kmer_length=model.kmer_length, site_count=site_count),
        SHMoofDataset(val_df, kmer_length=model.kmer_length, site_count=site_count),
        model,
        **burrito_params,
        name=trained_model_str(model_name, dataset_name),
    )

    if dataset_name == "tst":
        burrito.train(epochs=2)
    else:
        burrito.train(epochs=epochs)

    burrito.save_crepe(out_prefix)

    return model


def validation_burrito_of(crepe_prefix, dataset_name):
    crepe = load_crepe(crepe_prefix)
    model = crepe.model
    if dataset_name.startswith("shmoof_"):
        _, val_nickname = dataset_name.split("_")
        _, pcp_df = load_shmoof_dataframes(shmoof_path, val_nickname=val_nickname)
    elif dataset_name == "tangshm1k":
        pcp_df = pcp_df_of_nickname("tangshm", sample_count=1000)
    else:
        pcp_df = pcp_df_of_nickname(dataset_name)
    return RSSHMBurrito(
        None,
        SHMoofDataset(pcp_df, kmer_length=model.kmer_length, site_count=site_count),
        model,
        **burrito_params,
    )


def write_test_accuracy(crepe_prefix, dataset_name, directory="."):
    val_burrito = validation_burrito_of(crepe_prefix, dataset_name)
    bce_loss, csp_loss = val_burrito.evaluate()
    crepe_basename = os.path.basename(crepe_prefix)
    df = pd.DataFrame(
        {
            "crepe_prefix": [crepe_prefix],
            "crepe_basename": [crepe_basename],
            "dataset_name": [dataset_name],
            "bce_loss": [bce_loss.item()],
            "csp_loss": [csp_loss.item()],
        }
    )
    df.to_csv(
        f"{directory}/{crepe_basename}-ON-{dataset_name}.csv",
        index=False,
    )
