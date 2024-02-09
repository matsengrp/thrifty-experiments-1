import os

import pandas as pd

from netam import framework


dataset_dict = {
    "shmoof": "data/shmoof_pcp_2023-11-30_MASKED.csv",
    "tangshm": "data/tang-deepshm_size2_edges_22-May-2023.branch_length.csv",
    "cui": "data/cui-et-al-oof_pcp_2024-02-07_MASKED_NI.csv",
}

def localify(path):
    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, path)

dataset_dict = {name: localify(path) for name, path in dataset_dict.items()}


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
    pcp_df = pcp_df[pcp_df["parent"] != pcp_df["child"]]
    if sample_count is not None:
        pcp_df = pcp_df.sample(sample_count)
    pcp_df = pcp_df.reset_index(drop=True)

    return pcp_df


def train_val_dfs_of_nickname(dataset_name):
    """
    Returns the train and validation dataframes for the given dataset_name.
    
    What's a little confusing here is that some of the datasets are only used
    for test, so this function will return None for the train_df in those cases.
    """
    if dataset_name == "cui": 
        full_df = pcp_df_of_non_shmoof_nickname("cui")
        val_df = full_df[full_df["sample_id"] == "NP+GC1_BC9_IGK_Export_2017-02-02"]
        train_df = full_df.drop(val_df.index)
        return train_df, val_df
    elif dataset_name == "tangshm1k":
        val_df = pcp_df_of_non_shmoof_nickname("tangshm", sample_count=1000)
        return None, val_df
    elif dataset_name == "tangshm":
        val_df = pcp_df_of_non_shmoof_nickname("tangshm")
        return None, val_df
    # else we are doing a shmoof dataset
    if dataset_name == "tst":
        sample_count = 1000
        val_nickname = "small"
    else:
        sample_count = None
        shmoof, val_nickname = dataset_name.split("_")
        assert shmoof == "shmoof"
    train_df, val_df = load_shmoof_dataframes(
        dataset_dict["shmoof"], sample_count=sample_count, val_nickname=val_nickname
    )
    return train_df, val_df
