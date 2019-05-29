import os
import argparse
from tqdm import tqdm
import pandas as pd


# DATA_ROOT = 'dataset'
# MSET = 'train'
# TRAIN_CSV_NAME = 'train.csv'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='collect face image info to csv')
    parser.add_argument('--data_root', type=str, default='', help='directory to store face images')
    parser.add_argument('--mset', type=str, default='train', help='train/test mset')
    parser.add_argument('--ignore_unknown', default=False, action='store_true', help='include unknown class or not')
    parser.add_argument('--output', type=str, default='train.csv', help='train.csv / test.csv')
    args = parser.parse_args()

    DATA_PATH = '{}/{}'.format(args.data_root, args.mset)
    if not os.path.isdir(DATA_PATH):
        print('"{}" not found!'.format(DATA_PATH))
        exit(-1)
    # print(args.ignore_unknown)

    im_files = []
    im_labels = []

    # r=root, d=directory, f=files
    for r, d, _ in os.walk(DATA_PATH):
        if args.ignore_unknown:
            i = 0
        else:
            i = 1
        for SUBJECT in d:
            if args.ignore_unknown:
                if SUBJECT == 'unknown':
                    continue
            print('[INFO] Subject:', SUBJECT)
            sub_dir = '{}/{}'.format(r, SUBJECT)
            # print(sub_dir)
            for sr, _, files in os.walk(sub_dir):
                for f in tqdm(files):
                    im_files.append(SUBJECT + '_' + f)
                    # im_subjects.append(SUBJECT)
                    im_labels.append(0 if SUBJECT == 'unknown' else i)
            i += 1

    df = pd.DataFrame(list(zip(im_files, im_labels)), columns=['image', 'label'])
    df.to_csv(args.output, index=None)
