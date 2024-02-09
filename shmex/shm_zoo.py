import os
import sys

import numpy as np
import pandas as pd
from sklearn import metrics

import torch

torch.set_num_threads(1)

from netam.common import (
    mask_tensor_of,
    pick_device,
    parameter_count_of_model,
)
from netam.framework import (
    create_mutation_and_base_indicators,
    load_crepe,
    trimmed_shm_model_outputs_of_crepe,
    SHMoofDataset,
    RSSHMBurrito,
)
from netam import models

from shmple.evaluate import r_precision

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


def trained_model_str(model_name, data_nickname):
    return f"{model_name}-{data_nickname}"


def trained_model_path(model_name, data_nickname):
    return f"trained_models/{trained_model_str(model_name, data_nickname)}"


def train_model(model_name, dataset_name, resume=True):
    train_df, val_df = train_val_dfs_of_nickname(dataset_name)
    out_prefix = trained_model_path(model_name, dataset_name)
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

# TODO move this to a shm_evaluation or something


# TODO shouldn't base_indicator actually be base_idx everywhere?
def ragged_np_pcp_info(parents, children):
    mutation_indicator_list = []
    base_indicator_list = []
    mask_list = []
    for parent, child in zip(parents, children):
        mutation_indicators, base_indicators = create_mutation_and_base_indicators(
            parent, child
        )
        mutation_indicator_list.append(mutation_indicators.numpy())
        base_indicator_list.append(base_indicators.numpy())
        mask_list.append(mask_tensor_of(parent).numpy())
    return mutation_indicator_list, base_indicator_list, mask_list


def mut_accuracy_stats(mutation_indicator_list, rates_list, mask_list):
    mut_freqs = [
        indic.sum() / mask.sum()
        for indic, mask in zip(mutation_indicator_list, mask_list)
    ]
    rates_list = [
        rates[: len(indicator)] * mut_freq
        for indicator, rates, mut_freq in zip(
            mutation_indicator_list, rates_list, mut_freqs
        )
    ]
    rates_list = [rates[mask] for rates, mask in zip(rates_list, mask_list)]
    mutation_indicator_list = [
        indicator[mask] for indicator, mask in zip(mutation_indicator_list, mask_list)
    ]
    all_mutabilities = np.concatenate(rates_list)
    all_indicators = np.concatenate(mutation_indicator_list)
    return {
        "AUROC": metrics.roc_auc_score(all_indicators, all_mutabilities),
        "AUPRC": metrics.average_precision_score(all_indicators, all_mutabilities),
        "r-prec": r_precision(mutation_indicator_list, rates_list),
    }


def base_accuracy_stats(base_indicator_list, csp_list):
    filtered_base_indicators = np.concatenate(
        [indicator[indicator != -1] for indicator in base_indicator_list]
    )
    filtered_csps = np.concatenate(
        [csp[indicator != -1] for indicator, csp in zip(base_indicator_list, csp_list)]
    )
    all_predictions = filtered_csps.argmax(axis=-1)
    return {"sub_acc": (filtered_base_indicators == all_predictions).mean()}


def write_test_accuracy(crepe_prefix, dataset_name, directory="."):
    crepe_basename = os.path.basename(crepe_prefix)
    crepe = load_crepe(crepe_prefix)
    _, pcp_df = train_val_dfs_of_nickname(dataset_name)
    rates, csps = trimmed_shm_model_outputs_of_crepe(crepe, pcp_df["parent"])
    mut_indicators, base_indicators, masks = ragged_np_pcp_info(
        pcp_df["parent"], pcp_df["child"]
    )
    df_dict = {
        "crepe_prefix": crepe_prefix,
        "crepe_basename": crepe_basename,
        "parameter_count": parameter_count_of_model(crepe.model),
        "dataset_name": dataset_name,
    }
    df_dict.update(mut_accuracy_stats(mut_indicators, rates, masks))
    df_dict.update(base_accuracy_stats(base_indicators, csps))
    df = pd.DataFrame(df_dict, index=[0])
    df.to_csv(
        f"{directory}/{crepe_basename}-ON-{dataset_name}.csv",
        index=False,
    )
