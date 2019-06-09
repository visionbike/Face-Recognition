from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.models import Model, load_model
import imgaug as ia
from skimage import io
from skimage import transform
import face_alignment
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Dropout


# Killing optional CPU driver warnings
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

INPUT_DIMS = 128
NUM_CLASSES = 5
CSV_TEST_FILE_NAME = 'test.csv'

FACE_ALIGNER = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType._2D, flip_input=False, device='cpu')


def alignment(im, landmarks_points, width, height):
    if width == 96 and height == 96:
        src = np.array([
            [30.2946, 43.6963],
            [65.5318, 43.5014],
            [48.0252, 63.7366],
            [33.54993, 84.3655],
            [62.7299, 84.2041]], dtype=np.float32)
    elif width == 96 and height == 112:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
    elif width == 112 and height == 112:
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
    elif width == 150 and height == 150:
        src = np.array([
            [51.287415, 69.23612],
            [98.48009, 68.97509],
            [75.03375, 96.075806],
            [55.646385, 123.7038],
            [94.72754, 123.48763]], dtype=np.float32)
    elif width == 160 and height == 160:
        src = np.array([
            [54.706573, 73.85186],
            [105.045425, 73.573425],
            [80.036, 102.48086],
            [59.356144, 131.95071],
            [101.04271, 131.72014]], dtype=np.float32)
    elif width == 224 and height == 224:
        src = np.array([
            [76.589195, 103.3926],
            [147.0636, 103.0028],
            [112.0504, 143.4732],
            [83.098595, 184.731],
            [141.4598, 184.4082]], dtype=np.float32)
    else:
        return None
    trans = transform.SimilarityTransform()
    trans.estimate(landmarks_points, src)
    M = trans.params[0:2, :]
    face_im = cv2.warpAffine(im, M, (width, height), borderValue=0)
    return face_im


def pre_whitten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def get_y_true(df):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(row['label'])
    return np.array(y_true)


def Model():
    model = Sequential()
    model.add(Dense(256, input_dim=INPUT_DIMS, kernel_initializer='uniform', bias_initializer='uniform'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(256, kernel_initializer='uniform', bias_initializer='uniform'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(NUM_CLASSES, kernel_initializer='uniform', bias_initializer='uniform'))
    model.add(Activation('softmax'))
    return model


if __name__ == '__main__':
    test_df = pd.read_csv(CSV_TEST_FILE_NAME)

    y_true = get_y_true(test_df)
    print(y_true)

    nn4_small2_pretrained = load_model('facenet_notebook/nn4.small2.channel_first.h5')

    nn_model = Model()
    nn_model.summary()
    nn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    nn_model.load_weights('facenet_train_model/weights2/best_weight_part1_fold1.hdf5')

    acc_count = 0

    for i, fn in enumerate(test_df.image.values):
        print(fn)
        # read image
        im_path = 'dataset/test/{}'.format(fn)
        im = io.imread(im_path)

        aligned_faces = []

        # align face
        landmarks = FACE_ALIGNER.get_landmarks(im)
        if landmarks is None:
            for sigma in np.linspace(0., 3., num=11).tolist():
                seq = ia.augmenters.GaussianBlur(sigma)
                im_aug = seq.augment_image(im)
                landmarks = FACE_ALIGNER.get_landmarks(im_aug)
                if landmarks is not None:
                    # get each landmarks set
                    for points in landmarks:
                        # points = landmarks[0]
                        p1 = np.mean(points[36:42, :], axis=0)
                        p2 = np.mean(points[42:48, :], axis=0)
                        p3 = points[33, :]
                        p4 = points[48, :]
                        p5 = points[54, :]

                        if np.mean([p1[1], p2[1]]) < p3[1] < np.mean([p4[1], p5[1]]) \
                                and np.min([p4[1], p5[1]]) > np.max([p1[1], p2[1]]) \
                                and np.min([p1[1], p2[1]]) < p3[1] < np.max([p4[1], p5[1]]):
                            landmarks_points = np.array([p1, p2, p3, p4, p5], dtype=np.float32)

                            cv_im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                            aligned_face_im = alignment(cv_im, landmarks_points, 96, 96)
                            aligned_faces.append(cv2.imread(aligned_face_im, cv2.COLOR_BGR2RGB))
                            # del landmarks_points
        else:
            for points in landmarks:
                p1 = np.mean(points[36:42, :], axis=0)
                p2 = np.mean(points[42:48, :], axis=0)
                p3 = points[33, :]
                p4 = points[48, :]
                p5 = points[54, :]
                landmarks_points = np.array([p1, p2, p3, p4, p5], dtype=np.float32)

                cv_im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                aligned_face_im = alignment(cv_im, landmarks_points, 96, 96)
                aligned_faces.append(cv2.cvtColor(aligned_face_im, cv2.COLOR_BGR2RGB))
                # del landmarks_points

        if len(aligned_faces) == 0:
            continue

        for face_im in aligned_faces:
            norm_im = pre_whitten(face_im)
            norm_im = np.expand_dims(norm_im, axis=0)
            embedding = nn4_small2_pretrained.predict(norm_im)
            pred = nn_model.predict(embedding, batch_size=1, verbose=1)
            # print(pred)
            print('conf:', np.amax(pred, axis=1))
            pred_label = np.argmax(pred, axis=1)[0]
            print('predicted label:', pred_label)
            if y_true[i] == 'unknown':
                if pred_label == 0:
                    acc_count += 1
                continue
            elif int(y_true[i]) + 1 == pred_label:
                acc_count += 1

    print('acc:', acc_count / len(y_true))









