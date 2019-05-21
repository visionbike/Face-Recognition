import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from random import randint


FOLD_NUM = 5
SAMPLES_EACH_FOLD = 30
DATA_ROOT = 'dataset'
CSV_FILE_NAME = 'train.csv'
CSV_REFINED_FILE_NAME = 'train_refined.csv'

if __name__ == '__main__':
    df = pd.read_csv(CSV_FILE_NAME)
    df = df.sample(frac=1).reset_index(drop=True)
    indices = list(range(df.shape[0]))
    for i in range(SAMPLES_EACH_FOLD):
        info = {}
        rt = randint(1, 99)
        kf = KFold(n_splits=FOLD_NUM, random_state=rt, shuffle=True)
        for fold, (_, valid_index) in enumerate(kf.split(indices)):
            for vi in valid_index:
                info[vi] = fold
        myarr = []
        for idx in range(df.shape[0]):
            myarr.append(info[idx])
        df['rt{}'.format(i)] = np.array(myarr)
    df.to_csv(CSV_REFINED_FILE_NAME, index=False)