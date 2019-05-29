import os
import random
import threading
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Conv2D, Activation, BatchNormalization, Dense, Dropout, Flatten, Lambda, concatenate, add
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.python.keras.utils.np_utils import to_categorical
from clr_callback import CyclicLR

BATCH_SIZE = 32
EPOCHS = 200
NUM_FOLDS = 5
NUM_PARTS = 4
INPUT_DIMS = 512
NUM_CLASSES = 3
CSV_REFINED_FILE_NAME = 'train_ignore_unknown_refined.csv'


def train_generator(x_train_fold, y_train_fold, train_size, batch_size):
    while True:
        x_train_fold, y_train_fold = shuffle(x_train_fold, y_train_fold)
        for start in range(0, train_size, batch_size):
            end = min(start + batch_size, train_size)
            x_batch = np.array([], dtype=np.float32).reshape(0, 512)
            for i in range(start, end, 1):
                x_batch = np.vstack((x_batch, x_train_fold[i, random.randint(0, 99), :].reshape(1, 512)))
            y_batch = y_train_fold[start: end, :]
            yield x_batch, y_batch


def valid_generator(x_valid_fold, y_valid_fold, valid_size, batch_size):
    while True:
        for start in range(0, valid_size, batch_size):
            end = min(start + batch_size, valid_size)
            x_batch = x_valid_fold[start: end, :]
            y_batch = y_valid_fold[start: end, :]
            yield x_batch, y_batch


def get_y_true(df):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(to_categorical(row['label'], num_classes=NUM_CLASSES))
    return np.array(y_true)


def Model():
    model = Sequential()
    model.add(Dense(2048, input_dim=INPUT_DIMS, kernel_initializer='uniform', bias_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(NUM_CLASSES, kernel_initializer='uniform', bias_initializer='uniform'))
    model.add(Activation('softmax'))
    return model


if __name__ == '__main__':
    train_df = pd.read_csv('../{}'.format(CSV_REFINED_FILE_NAME))

    x_train = np.load('train_data.npy')
    x_train_aug = np.load('train_aug_data.npy')
    y_train = get_y_true(train_df)

    if not os.path.exists('weights'):
        os.makedirs('weights')

    train_log = open('train_log.txt', 'w')
    loss_average = 0.0
    acc_average = 0.0
    for part in random.sample(range(30), NUM_PARTS):
        for fold in range(NUM_FOLDS):
            v_df = train_df.loc[train_df['rt{}'.format(part)] == fold]
            v_idx = v_df.index.values.tolist()
            t_df = train_df.loc[~train_df.index.isin(v_df.index)]
            t_idx = t_df.index.values.tolist()

            print('[INFO] **************Part %d    Fold %d**************' % (part, fold))

            x_train_fold = x_train_aug[t_idx, :, :]
            y_train_fold = y_train[t_idx, :]

            x_valid_fold = x_train[v_idx, :]
            y_valid_fold = y_train[v_idx, :]

            train_size = len(t_idx)
            valid_size = len(v_idx)

            train_steps = int(np.ceil(float(train_size) / float(BATCH_SIZE)))
            valid_steps = int(np.ceil(float(valid_size) / float(BATCH_SIZE)))

            print('[INFO] TRAIN SIZE:%d VALID SIZE:%d' % (train_size, valid_size))

            WEIGHTS_BEST = 'weights/best_weight_part{}_fold{}.hdf5'.format(part, fold)

            clr = CyclicLR(base_lr=1e-7, max_lr=1e-3, step_size=6 * train_steps, mode='exp_range', gamma=0.99994)
            early_stopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='max')
            save_checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor='val_acc', verbose=1, save_weights_only=True,
                                              save_best_only=True, mode='max')
            callbacks = [save_checkpoint, early_stopping, clr]

            model = Model()
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

            model.fit_generator(generator=train_generator(x_train_fold, y_train_fold, train_size, BATCH_SIZE),
                                steps_per_epoch=train_steps, epochs=EPOCHS, verbose=1,
                                validation_data=valid_generator(x_valid_fold, y_valid_fold, valid_size, BATCH_SIZE),
                                validation_steps=valid_steps, callbacks=callbacks)

            model.load_weights(WEIGHTS_BEST)

            score = model.evaluate(x=x_valid_fold, y=y_valid_fold, batch_size=BATCH_SIZE, verbose=1)
            loss_average += score[0]
            acc_average += score[1]
            train_log.write('[INFO] PART:%d FOLD:%d LOSS:%f ACC:%f\n' % (part, fold, score[0], score[1]))

            K.clear_session()

    loss_average /= float(NUM_PARTS*NUM_FOLDS)
    acc_average /= float(NUM_PARTS*NUM_FOLDS)
    train_log.write('[INFO] AVERAGE LOSS:%f ACC:%f\n' % (loss_average, acc_average))
    train_log.close()
