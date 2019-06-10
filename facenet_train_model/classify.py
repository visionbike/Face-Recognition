from scipy import misc
import cv2
import imutils
import tensorflow as tf
import numpy as np
import os
import sys
import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Model, load_model
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from skimage import io
from skimage import transform
import face_alignment
from sklearn_porter import Porter
from prepare_data import prepare_data_

CSV_REFINED_FILE_NAME = 'train_refined.csv'
CSV_TEST_FILE_NAME = 'test-1.csv'
path_svm = 'weight_svm.pickle'
path_knn = 'weight_knn.pickle'
INPUT_DIMS = 128
NUM_CLASSES = 5

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

def get_y_train(df):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(to_categorical(row['label'], num_classes=NUM_CLASSES))
    return np.array(y_true)

def get_y_true(df):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(row['label'])
    return np.array(y_true)

def load_train_data(non_unknow = True):
    train_n = 'train_data.npy'
    aug_n = 'train_aug_data.npy'

    train_df = pd.read_csv('../{}'.format(CSV_REFINED_FILE_NAME))
    y_train = get_y_train(train_df)
    y_train = np.argmax(y_train, axis=1)
    if non_unknow == True:
        train_noun = train_df.loc[train_df['image'].str[:7] != 'unknown']
        t_idx = train_noun.index.values.tolist()
        y_train = y_train[t_idx] - 1

        train_n = 'train_data_noun.npy'
        aug_n = 'train_aug_data_noun.npy'

    x_train = np.load(train_n)
    x_train_ = x_train[:, np.newaxis, :]
    x_train_aug = np.load(aug_n)

    print(len(train_df), np.shape(x_train), np.shape(x_train_aug), np.shape(y_train))
    embeddings = np.concatenate((x_train_, x_train_aug), axis=1)

    embed = []
    for ind in range(len(embeddings[0, :, 0])):
        if len(embed) == 0:
            embed = embeddings[:, ind, :]
            arr_lb = y_train
        else:
            embed = np.vstack((embed, embeddings[:, ind, :]))
            arr_lb = np.hstack((arr_lb, y_train))

    print(np.shape(embeddings), np.shape(y_train), np.shape(embed), np.shape(arr_lb))
    return x_train, y_train, embed, arr_lb

def load_test_data(non_unknow = True):
    if non_unknow == True:
        embed_n = "test_emb_noun.npy"
        lable_n = "test_lab_noun.npy"
    else:
        embed_n = "test_emb.npy"
        lable_n = "test_lab.npy"

    embeddings = np.load(embed_n)
    y_true = np.load(lable_n)

    return embeddings, y_true

def embed_test_data(non_unknow = True):
    test_df = pd.read_csv('../{}'.format(CSV_TEST_FILE_NAME))
    y_true = get_y_true(test_df)

    print("embed_test_data", np.shape(y_true), len(test_df))
    if non_unknow == True:
        # test_df = test_df.loc[test_df['image'].str[:7] == 'unknown']
        test_df = test_df.loc[test_df['label'] != 0]
        t_idx = test_df.index.values.tolist()
        y_true = y_true[t_idx] - 1

    print(np.shape(y_true), len(test_df))

    nn4_small2_pretrained = load_model('../facenet_notebook/nn4.small2.channel_first.h5')
    embeddings = []
    for i, fn in enumerate(test_df.image.values):
        print(fn)
        # read image
        im_path = '../dataset/test/{}'.format(fn)
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
            embedding = embedding.reshape((INPUT_DIMS))

        embeddings.append(embedding)

    if non_unknow == True:
        embed_n = "test_emb_noun.npy"
        lable_n = "test_lab_noun.npy"
    else:
        embed_n = "test_emb.npy"
        lable_n = "test_lab.npy"

    np.save(embed_n, embeddings)
    np.save(lable_n, y_true)

    return embeddings, y_true


def train_knn(embeddings, labels, nlabels):
    uni, counts = np.unique(labels, return_counts=True)
    neigh = KNeighborsClassifier(n_neighbors=len(uni))
    neigh.fit(embeddings, labels)

    with open(path_knn, 'wb') as file:
        pickle.dump(neigh, file)

    porter = Porter(neigh, language='java')
    output = porter.export()

    with open('KNeighborsClassifier.java', 'w') as f:
        f.write(output)

    print(neigh.score(embeddings, labels))

def predict_knn(embeddings, nlabels=None):
    if os.path.isfile(path_knn) == False:
        print("Don't have file", path_knn)
        return 0

    print("Predict Knn:  ", end='')
    with open(path_knn, 'rb') as file:
        clf = pickle.load(file)
        
    cls_pre = clf.predict(embeddings)
    print(cls_pre)
    # cls_pre = np.argmax(cls_pre, axis=1)

    # return [nlabels[cls_] for cls_ in cls_pre]
    return cls_pre


def train_svm(embeddings, labels, nlabels):
    svm = SVC(gamma='auto')
    # labels = np.argmax(labels, axis=1)
    print(np.shape(embeddings), np.shape(labels))

    svm.fit(embeddings, labels)
    with open(path_svm, 'wb') as file:
        pickle.dump(svm, file)

    porter = Porter(svm, language='java')
    output = porter.export()

    with open('svm_SVC.java', 'w') as f:
        f.write(output)

    print(svm.score(embeddings, labels))

def predict_svm(embeddings, nlabels=None):
    if os.path.isfile(path_svm) == False:
        print("Don't have file", path_svm)
        return 0

    with open(path_svm, 'rb') as file:
        svm = pickle.load(file)

    cls_pre = svm.predict(embeddings)
    print(cls_pre)

    # return [nlabels[cls_] for cls_ in cls_pre]
    return cls_pre

non_unknow = True

prepare_data_(non_unknow)
nlabels = ['hank', 'pieter', 'phuc', 'palo', 'unknown']
x_embed, y_train, embed, arr_lb = load_train_data(non_unknow)
embeddings, y_true = embed_test_data(non_unknow)
embeddings, y_true = load_test_data(non_unknow)

run_knn = True
run_svm = True

if run_knn == True:
    print("\n Knn running")
    train_knn(x_embed, y_train, nlabels)
    pred = predict_knn(embeddings)
    print("true: ", y_true)
    print(accuracy_score(pred, y_true))

if run_svm == True:
    print("\n Svm running")
    train_svm(x_embed, y_train, nlabels)
    pred = predict_svm(embeddings)
    print("true: ", y_true)
    print(accuracy_score(pred, y_true))
