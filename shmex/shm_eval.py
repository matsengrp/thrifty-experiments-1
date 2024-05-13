import os
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from netam.common import (
    nt_mask_tensor_of,
    parameter_count_of_model,
)
from netam.framework import (
    encode_mut_pos_and_base,
    load_crepe,
    trimmed_shm_model_outputs_of_crepe,
)
from epam import evaluation

sys.path.append("..")
from shmex.shm_data import train_val_dfs_of_nickname
from shmex.shm_zoo import standardize_and_optimize_branch_lengths

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


def ragged_np_pcp_encoding(parents, children, site_count=None):
    """
    Encode the mutation indicators, base indices, and masks of a list of
    parent-child pairs in a way that can be used to calculate the accuracy of
    the model's predictions using sklearn.metrics.
    """
    mutation_indicator_list = []
    base_idxs_list = []
    mask_list = []
    for parent, child in zip(parents, children):
        mutation_indicators, base_idxs = encode_mut_pos_and_base(parent, child)
        mutation_indicator_list.append(mutation_indicators.numpy()[:site_count])
        base_idxs_list.append(base_idxs.numpy()[:site_count])
        mask_list.append(nt_mask_tensor_of(parent).numpy()[:site_count])
    return mutation_indicator_list, base_idxs_list, mask_list


def reset_outside_of_shmoof_region_single(indicator, reset_value):
    indicator = indicator.copy()
    indicator[:80] = reset_value
    indicator[320:] = reset_value
    return indicator

    
def reset_outside_of_shmoof_region(indicators, reset_value):
    return [reset_outside_of_shmoof_region_single(indicator, reset_value) for indicator in indicators]


def mut_accuracy_stats(mutation_indicator_list, rates_list, bl_array, mask_list):
    assert len(mutation_indicator_list) == len(rates_list) == len(bl_array) == len(mask_list)
    rates_list = [
        rates[: len(indicator)] * bl
        for indicator, rates, bl in zip(
            mutation_indicator_list, rates_list, bl_array
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
        "mut_pos_ll": - metrics.log_loss(
            all_indicators, all_mutabilities, labels=[0, 1]
        ),
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
    cat_log_like = - metrics.log_loss(true_labels_one_hot, filtered_csp_arr)

    return {"sub_acc": accuracy, "base_ll": cat_log_like}


def oe_plot_of(ratess, masks, branch_lengths, mut_indicators, suptitle_prefix=""):
    mut_probs_l = []
    mut_indicators_l = []

    for rates, mask, branch_length, mut_indicator in zip(ratess, masks, branch_lengths, mut_indicators):
        mut_probs = 1.0 - torch.exp(-rates * branch_length)
        mut_probs_l.append(mut_probs[mask])
        mut_indicators_l.append(mut_indicator[mask])

    oe_plot_df = pd.DataFrame({
        "prob": torch.cat(mut_probs_l).numpy(),
        "mutation": np.concatenate(mut_indicators_l),
    })

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    result_dict = evaluation.plot_observed_vs_expected(oe_plot_df, None, axs[0], axs[1])
    if suptitle_prefix != "":
        suptitle_prefix = suptitle_prefix + "; "
    fig.suptitle(f"{suptitle_prefix}overlap={result_dict['overlap']:.3g}, residual={result_dict['residual']:.3g}", fontsize=16)
    plt.tight_layout()

    return fig, result_dict


def write_test_accuracy(crepe_prefix, dataset_name, directory=".", restrict_evaluation_to_shmoof_region=False):
    crepe_basename = os.path.basename(crepe_prefix)
    crepe = load_crepe(crepe_prefix)
    _, pcp_df = train_val_dfs_of_nickname(dataset_name)
    ratess, cspss = trimmed_shm_model_outputs_of_crepe(crepe, pcp_df["parent"])
    site_count = crepe.encoder.site_count
    mut_indicators, base_idxss, masks = ragged_np_pcp_encoding(pcp_df["parent"], pcp_df["child"], site_count)
    standardize_and_optimize_branch_lengths(crepe.model, pcp_df)
    val_bls = pcp_df["branch_length"].values
    if restrict_evaluation_to_shmoof_region:
        mut_indicators = reset_outside_of_shmoof_region(mut_indicators, 0)
        base_idxss = reset_outside_of_shmoof_region(base_idxss, -1)
        masks = reset_outside_of_shmoof_region(masks, 0)
    df_dict = {
        "crepe_prefix": crepe_prefix,
        "crepe_basename": crepe_basename,
        "parameter_count": parameter_count_of_model(crepe.model),
        "dataset_name": dataset_name,
    }
    df_dict.update(mut_accuracy_stats(mut_indicators, ratess, val_bls, masks))
    df_dict.update(base_accuracy_stats(base_idxss, cspss))
    comparison_title = f"{crepe_basename}-ON-{dataset_name}"
    fig, oe_results = oe_plot_of(ratess, masks, val_bls, mut_indicators, comparison_title)
    oe_results.pop("counts_twinx_ax")
    df_dict.update(oe_results)
    df = pd.DataFrame(df_dict, index=[0])
    df.to_csv(
        f"{directory}/{comparison_title}.csv",
        index=False,
    )
    fig.savefig(f"{directory}/{comparison_title}.png")
