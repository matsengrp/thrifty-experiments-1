import os
import re

import pandas as pd

from netam import framework


dataset_dict = {
    "shmoof": "data/v0/shmoof_pcp_2023-11-30_MASKED.csv.gz",
    "tangshm": "data/v1/tang-deepshm-oof_pcp_2024-04-09_MASKED_NI.csv.gz",
    "cui": "data/v0/cui-et-al-oof_pcp_2024-02-22_MASKED_NI.csv.gz",
    "cuims": "data/v0/cui-et-al-oof-msproc_pcp_2024-02-29_MASKED_NI.csv",
    "greiff": "data/v0/greiff-systems-oof_pcp_2023-11-30_MASKED_NI.csv.gz",
    "syn10x": "data/v1/wyatt-10x-1p5m_fs-all_pcp_2024-04-29_NI_SYN.csv.gz",
    "oracleshmoofcnn10k": "data/v0/mimic_shmoof_CNNJoiLrgShmoofSmall.10K.csv.gz",
    "oracletangcnn": "data/v0/mimic_tang_CNNJoiLrgShmoofSmall.csv.gz",
}


def localify(path):
    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, path)


dataset_dict = {name: localify(path) for name, path in dataset_dict.items()}


def parent_and_child_differ(row):
    for p, c in zip(row["parent"], row["child"]):
        if c != "N" and p != c and p != "N":
            return True
    return False


def load_shmoof_dataframes(
    csv_path, sample_count=None, val_nickname="13", random_seed=42
):
    """Load the shmoof dataframes from the csv_path and return train and validation dataframes.

    Args:
        csv_path (str): Path to the csv file containing the shmoof data.
        sample_count (int, optional): Number of samples to use. Defaults to None.
        val_nickname (str, optional): Nickname of the sample to use for validation. Defaults to "13".

    Returns:
        tuple: Tuple of train and validation dataframes.

    Notes:

    The sample nicknames are: `51` is the biggest one, `13` is the second biggest,
    and `small` is the rest of the repertoires merged together.

    If the nickname is `split`, then we do a random 80/20 split of the data.

    Here are the value_counts:
    51       22424
    13       13186
    59        4686
    88        3067
    97        3028
    small     2625
    """
    full_shmoof_df = pd.read_csv(csv_path, index_col=0).reset_index(drop=True)

    # only keep rows where parent is different than child
    full_shmoof_df = full_shmoof_df[full_shmoof_df["parent"] != full_shmoof_df["child"]]

    if sample_count is not None:
        full_shmoof_df = full_shmoof_df.sample(sample_count)

    if val_nickname == "split":
        train_df = full_shmoof_df.sample(frac=0.8, random_state=random_seed)
        val_df = full_shmoof_df.drop(train_df.index)
        return train_df, val_df

    # else
    full_shmoof_df["nickname"] = full_shmoof_df["sample_id"].astype(str).str[-2:]
    for small_nickname in ["80", "37", "50", "07"]:
        full_shmoof_df.loc[
            full_shmoof_df["nickname"] == small_nickname, "nickname"
        ] = "small"

    val_df = full_shmoof_df[full_shmoof_df["nickname"] == val_nickname]
    train_df = full_shmoof_df.drop(val_df.index)

    assert len(val_df) > 0, f"No validation samples found with nickname {val_nickname}"

    return train_df, val_df


def pcp_df_of_non_shmoof_nickname(dataset_name, sample_count=None):
    print(f"Loading {dataset_dict[dataset_name]}")

    pcp_df = pd.read_csv(dataset_dict[dataset_name], index_col=0)
    pcp_df = pcp_df[pcp_df.apply(parent_and_child_differ, axis=1)]
    if sample_count is not None:
        pcp_df = pcp_df.sample(sample_count)
    pcp_df = pcp_df.reset_index(drop=True)

    return pcp_df


def pcp_df_of_non_shmoof_nickname_using_k_for_sample_count(dataset_name):
    """
    If dataset_name ends with "Yk" where Y is a number, then we take that number
    and call pcp_df_of_non_shmoof_nickname with that number of samples.
    """
    if dataset_name[-1] == "k":
        regex = r"(.*[^\d])(\d+)k"
        dataset_name, sample_count = re.match(regex, dataset_name).groups()
        sample_count = int(sample_count) * 1000
    else:
        sample_count = None
    return pcp_df_of_non_shmoof_nickname(dataset_name, sample_count)


def train_val_split_from_val_sample_ids(full_df, val_sample_ids):
    """
    Splits the full_df into train and validation dataframes based on the val_sample_ids.
    """
    val_df = full_df[full_df["sample_id"].isin(val_sample_ids)]
    train_df = full_df.drop(val_df.index)
    return train_df, val_df


def train_val_dfs_of_nickname(dataset_name):
    """
    Returns the train and validation dataframes for the given dataset_name.

    What's a little confusing here is that some of the datasets are only used
    for test, so this function will return None for the train_df in those cases.

    When dataset_name starts with "val_", then we return the whole dataset as
    the validation set.
    """
    if dataset_name == "cui":
        full_df = pcp_df_of_non_shmoof_nickname("cui")
        val_df = full_df[full_df["sample_id"] == "NP+GC1_BC9_IGK_Export_2017-02-02"]
        train_df = full_df.drop(val_df.index)
        return train_df, val_df
    # TODO
    elif dataset_name == "cui_overtrain":
        full_df = pcp_df_of_non_shmoof_nickname("cui")
        return full_df, full_df.copy()
    elif dataset_name == "greiff":
        full_df = pcp_df_of_non_shmoof_nickname("greiff")
        val_sample_ids = [
            "no-vax_m5_plasma",
            "ova-vax_m1_plasma",
            "hepb-vax_m2_plasma",
            "np-hel-vax_m4_plasma",
        ]
        return train_val_split_from_val_sample_ids(full_df, val_sample_ids)
    elif dataset_name == "val_cui":
        val_df = pcp_df_of_non_shmoof_nickname("cui")
        return None, val_df
    elif dataset_name == "val_tangshm1k":
        val_df = pcp_df_of_non_shmoof_nickname("tangshm", sample_count=1000)
        return None, val_df
    elif dataset_name == "val_tangshm":
        val_df = pcp_df_of_non_shmoof_nickname("tangshm")
        return None, val_df
    elif dataset_name.startswith("syn10x"):
        full_df = pcp_df_of_non_shmoof_nickname_using_k_for_sample_count(dataset_name)
        val_sample_ids = ["d4"] # this one has about 25% of the data
        return train_val_split_from_val_sample_ids(full_df, val_sample_ids)
    elif dataset_name.startswith("val_syn10x"):
        val_df = pcp_df_of_non_shmoof_nickname_using_k_for_sample_count(dataset_name)
        return None, val_df
    elif dataset_name == "val_oracleshmoofcnn10k":
        val_df = pcp_df_of_non_shmoof_nickname("oracleshmoofcnn10k")
        return None, val_df
    elif dataset_name == "val_oracletangcnn":
        val_df = pcp_df_of_non_shmoof_nickname("oracletangcnn")
        return None, val_df
     # else we are doing a shmoof dataset
    if dataset_name == "tst":
        sample_count = 1000
        val_nickname = "small"
    else:
        sample_count = None
        shmoof, val_nickname = dataset_name.split("_")
        assert shmoof == "shmoof", f"Dataset {dataset_name} not recognized"
    train_df, val_df = load_shmoof_dataframes(
        dataset_dict["shmoof"], sample_count=sample_count, val_nickname=val_nickname
    )
    return train_df, val_df


def train_val_dfs_of_nicknames(dataset_names):
    """
    Splits dataset_names by "+", runs train_val_dfs_of_nickname on each one,
    and combines each pair of train and validation dataframes into a single
    pair of train and validation dataframes.
    """
    dataset_names = dataset_names.split("+")
    train_dfs = []
    val_dfs = []
    for dataset_name in dataset_names:
        train_df, val_df = train_val_dfs_of_nickname(dataset_name)
        if train_df is not None:
            train_dfs.append(train_df)
        val_dfs.append(val_df)
    def concat_dfs(dfs):
        if len(dfs) == 0:
            return None
        return pd.concat(dfs).reset_index(drop=True)
    return tuple([concat_dfs(dfs) for dfs in [train_dfs, val_dfs]])