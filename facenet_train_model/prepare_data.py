import os
from shutil import copyfile
from multiprocessing import Pool
import pandas as pd
import numpy as np
from tqdm import *

embed_dims = 128
MODEL_NAME = 'nn4_small2'
CSV_REFINED_FILE_NAME = 'train_refined.csv'


def prepare_train_data(file_name):
    name = file_name.split('.')[0]
    emb_path = '../models/facenet/embeddings/{}/train/{}'.format(MODEL_NAME, name + '.npy')
    emb = np.load(emb_path).reshape(embed_dims)
    return emb


def prepare_train_aug_data(file_name):
    name = file_name.split('.')[0]
    emb_path = '../models/facenet/embeddings/{}/train/{}'.format(MODEL_NAME, name + '_augmentation.npy')
    emb = np.load(emb_path).reshape(100, embed_dims)
    return emb


if __name__ == '__main__':
    train_df = pd.read_csv('../{}'.format(CSV_REFINED_FILE_NAME))

    # prepare train data
    print('[INFO] Prepare train data...')
    p = Pool(8)
    train_data = p.map(func=prepare_train_data, iterable=train_df.image.values.tolist())
    p.close()
    train_data = np.array(train_data)
    print('[INFO] Shape:', train_data.shape)
    np.save('train_data.npy', train_data)
    train_data = []

    # prepare augmented train data
    print('[INFO] Prepare augmented train data...')
    p = Pool(8)
    train_aug_data = p.map(func=prepare_train_aug_data, iterable=train_df.image.values.tolist())
    p.close()
    train_aug_data = np.array(train_aug_data)
    print('[INFO] Shape:', train_aug_data.shape)
    np.save('train_aug_data.npy', train_aug_data)
    train_aug_data = []
