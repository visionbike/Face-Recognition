#!/usr/bin/env bash
export PATH=$PATH:$(pwd)
# create train.csv
python create_dataset_csv.py --data_root dataset --mset train --output train.csv

