#!/bin/bash

# This script copies the trained models with their appropriate names to the thrifty-models directory.

BASE_DEST_DIR="../../thrifty-models/models"

# Re "_notbig" below, note that because `Snakefile_train_on_all_data` uses val_is_train=True, it's actually training on the full dataset.
file_pairs=(
    "cnn_ind_lrg-shmoof_notbig+tangshm-simple-0 TH1-20"
    "cnn_ind_med-shmoof_notbig+tangshm-simple-0 TH1-45"
    "cnn_joi_lrg-shmoof_notbig+tangshm-simple-0 TH1-59"
)

for pair in "${file_pairs[@]}"; do
    src_base=$(echo $pair | awk '{print $1}')
    dest_base=$(echo $pair | awk '{print $2}')
    cp "trained_models/${src_base}.pth" "$BASE_DEST_DIR/${dest_base}.pth"
    cp "trained_models/${src_base}.yml" "$BASE_DEST_DIR/${dest_base}.yml"
done
