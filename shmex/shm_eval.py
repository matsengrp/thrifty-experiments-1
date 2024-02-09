import os
import sys

import numpy as np
import pandas as pd
from sklearn import metrics

from netam.common import (
    mask_tensor_of,
    parameter_count_of_model,
)
from netam.framework import (
    encode_mut_pos_and_base,
    load_crepe,
    trimmed_shm_model_outputs_of_crepe,
)

sys.path.append("..")
from shmex.shm_data import train_val_dfs_of_nickname

from shmple.evaluate import r_precision


def ragged_np_pcp_encoding(parents, children):
    mutation_indicator_list = []
    base_idxs_list = []
    mask_list = []
    for parent, child in zip(parents, children):
        mutation_indicators, base_idxs = encode_mut_pos_and_base(
            parent, child
        )
        mutation_indicator_list.append(mutation_indicators.numpy())
        base_idxs_list.append(base_idxs.numpy())
        mask_list.append(mask_tensor_of(parent).numpy())
    return mutation_indicator_list, base_idxs_list, mask_list


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


def base_accuracy_stats(base_idxs_list, csp_list):
    filtered_base_idxs_list = np.concatenate(
        [indicator[indicator != -1] for indicator in base_idxs_list]
    )
    filtered_csp_list = np.concatenate(
        [csp[indicator != -1] for indicator, csp in zip(base_idxs_list, csp_list)]
    )
    all_predictions = filtered_csp_list.argmax(axis=-1)
    return {"sub_acc": (filtered_base_idxs_list == all_predictions).mean()}


def write_test_accuracy(crepe_prefix, dataset_name, directory="."):
    crepe_basename = os.path.basename(crepe_prefix)
    crepe = load_crepe(crepe_prefix)
    _, pcp_df = train_val_dfs_of_nickname(dataset_name)
    rates, csps = trimmed_shm_model_outputs_of_crepe(crepe, pcp_df["parent"])
    mut_indicators, base_idxs, masks = ragged_np_pcp_encoding(
        pcp_df["parent"], pcp_df["child"]
    )
    df_dict = {
        "crepe_prefix": crepe_prefix,
        "crepe_basename": crepe_basename,
        "parameter_count": parameter_count_of_model(crepe.model),
        "dataset_name": dataset_name,
    }
    df_dict.update(mut_accuracy_stats(mut_indicators, rates, masks))
    df_dict.update(base_accuracy_stats(base_idxs, csps))
    df = pd.DataFrame(df_dict, index=[0])
    df.to_csv(
        f"{directory}/{crepe_basename}-ON-{dataset_name}.csv",
        index=False,
    )