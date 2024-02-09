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


# Taken from shmple.
def r_precision(y_true: list[np.ndarray], y_pred: list[np.ndarray]):

    ret = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        num_muts = t.sum()
        if num_muts > 0:
            assert len(p) == len(t)
            total_vals = len(p)
            idxs = np.argpartition(p, kth=total_vals - num_muts)[-num_muts:]
            ret.append(np.array(t)[idxs].sum() / num_muts)
    if len(ret) > 0:
        return sum(ret) / len(ret)
    else:
        return np.array([0.0])


def ragged_np_pcp_encoding(parents, children):
    mutation_indicator_list = []
    base_idxs_list = []
    mask_list = []
    for parent, child in zip(parents, children):
        mutation_indicators, base_idxs = encode_mut_pos_and_base(parent, child)
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
        "mut_pos_xent": metrics.log_loss(all_indicators, all_mutabilities, labels=[0, 1]),
    }


def base_accuracy_stats(base_idxs_list, csp_list):
    filtered_base_idxs_arr = np.concatenate(
        [indicator[indicator != -1] for indicator in base_idxs_list]
    )
    filtered_csp_arr = np.concatenate(
        [csp[indicator != -1] for indicator, csp in zip(base_idxs_list, csp_list)]
    )
    
    all_predictions = filtered_csp_arr.argmax(axis=-1)
    accuracy = (filtered_base_idxs_arr == all_predictions).mean()
    
    # Prepare the true labels in the format expected by log_loss: one-hot encoded vectors
    # Since filtered_base_idxs_list contains class indices from 0 to 3, use them to create one-hot encodings
    num_classes = 4
    true_labels_one_hot = np.eye(num_classes)[filtered_base_idxs_arr.astype(int)]
    cat_cross_entropy = metrics.log_loss(true_labels_one_hot, filtered_csp_arr)
    
    return {"sub_acc": accuracy, "base_xent": cat_cross_entropy}


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
