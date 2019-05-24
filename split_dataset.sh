#!/usr/bin/env bash

export PATH=$PATH:$(pwd)
# train_refined.csv
python split_dataset.py --csv_file train.csv --n_fold 5 --samples_per_fold 30 --output train_refined.csv
