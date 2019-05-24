import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from random import randint


# FOLD_NUM = 5
# SAMPLES_EACH_FOLD = 30
# CSV_FILE_NAME = 'train.csv'
# CSV_REFINED_FILE_NAME = 'train_refined.csv'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='train.csv', help='train csv file')
    parser.add_argument('--n_fold', type=int, default=5, help='number of folds')
    parser.add_argument('--samples_per_fold', type=int, default=30, help='number of samples per fold')
    parser.add_argument('--output', type=str, default='train_refined.csv', help='refined csv file')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    df = df.sample(frac=1).reset_index(drop=True)
    indices = list(range(df.shape[0]))
    for i in range(args.samples_per_fold):
        info = {}
        rt = randint(1, 99)
        kf = KFold(n_splits=args.n_fold, random_state=rt, shuffle=True)
        for fold, (_, valid_index) in enumerate(kf.split(indices)):
            for vi in valid_index:
                info[vi] = fold
        myarr = []
        for idx in range(df.shape[0]):
            myarr.append(info[idx])
        df['rt{}'.format(i)] = np.array(myarr)
    df.to_csv(args.output, index=False)
