import os

import pandas as pd

from netam import framework


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
    # TODO temporary
    # full_shmoof_df = full_shmoof_df[full_shmoof_df["parent"] != full_shmoof_df["child"]]

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


def pcp_df_of_nickname(dataset_name, sample_count=None):
    dataset_dict = {
        "tangshm": "data/tang-deepshm_size2_edges_22-May-2023.branch_length.csv"
    }

    def localify(path):
        home_dir = os.path.expanduser("~")
        return os.path.join(home_dir, path)

    dataset_dict = {name: localify(path) for name, path in dataset_dict.items()}
    print(f"Loading {dataset_dict[dataset_name]}")

    pcp_df = pd.read_csv(dataset_dict[dataset_name], index_col=0)
    pcp_df = pcp_df[pcp_df["parent"] != pcp_df["child"]]
    if sample_count is not None:
        pcp_df = pcp_df.sample(sample_count)
    pcp_df = pcp_df.reset_index(drop=True)

    return pcp_df
