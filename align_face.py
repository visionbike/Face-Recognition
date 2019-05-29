import os
from tqdm import tqdm
import cv2
import numpy as np
import imgaug as ia
from skimage import io
from skimage import transform
import face_alignment

from helper import make_if_not_exist

verbose = False

DATA_ROOT = 'dataset'
# SUBJECT = 'phuc'
MSET = 'train'  # train/test
SCALES = [[96, 112], [112, 112], [150, 150], [160, 160], [224, 224]]

FACE_ALIGNER = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType._2D, flip_input=False,
                                            device='cpu')


def alignment(im, landmarks_points, width, height):
    if width == 96 and height == 112:
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


if __name__ == '__main__':
    DATA_DIR = '{}/{}'.format(DATA_ROOT, MSET)
    print('[INFO] Creating aligned image directory...')
    for SUBJECT in os.listdir(DATA_DIR):
        for scale in SCALES:
            ALIGNED_DIR = '{}/aligned/{}/{}/{}x{}'.format(DATA_ROOT, MSET, SUBJECT, scale[0], scale[1])
            make_if_not_exist(ALIGNED_DIR)

    print('[INFO] Detecting and aligning faces...')
    for SUBJECT in os.listdir(DATA_DIR):
        SUBJECT_DATA_DIR = '{}/{}'.format(DATA_DIR, SUBJECT)
        print('[INFO] Subject: {}'.format(SUBJECT))
        for rdir, _, files in os.walk(SUBJECT_DATA_DIR):
            for file in tqdm(files):
                im_path = os.path.join(rdir, file)
                im = io.imread(im_path)
                landmarks = FACE_ALIGNER.get_landmarks(im)
                if landmarks is None:
                    print('[INFO] Unknown faces in "{}"'.format(im_path))
                    print('[INFO] Pre-processing image...')
                    for sigma in np.linspace(0., 3., num=11).tolist():
                        seq = ia.augmenters.GaussianBlur(sigma)
                        im_aug = seq.augment_image(im)
                        landmarks = FACE_ALIGNER.get_landmarks(im_aug)
                        if landmarks is not None:
                            print('[INFO] sigma:', sigma)
                            points = landmarks[0]
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

                                if verbose:
                                    for point in points:
                                        cv2.circle(cv_im, tuple(point), 2, (255, 255, 255), 2, cv2.LINE_AA)
                                    cv2.imshow('im', cv_im)
                                    cv2.waitKey(0)

                                for scale in SCALES:
                                    aligned_face_im = alignment(cv_im, landmarks_points, scale[0], scale[1])

                                    if verbose:
                                        cv2.imshow('face_im', aligned_face_im)
                                        cv2.waitKey(2)

                                    cv2.imwrite('{}/aligned/{}/{}/{}x{}/{}'.format(DATA_ROOT,
                                                                                   MSET,
                                                                                   SUBJECT,
                                                                                   scale[0],
                                                                                   scale[1],
                                                                                   file),
                                                aligned_face_im)
                                break

                else:
                    points = landmarks[0]
                    p1 = np.mean(points[36:42, :], axis=0)
                    p2 = np.mean(points[42:48, :], axis=0)
                    p3 = points[33, :]
                    p4 = points[48, :]
                    p5 = points[54, :]
                    landmarks_points = np.array([p1, p2, p3, p4, p5], dtype=np.float32)

                    cv_im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                    if verbose:
                        for point in points:
                            cv2.circle(cv_im, tuple(point), 2, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow('im', cv_im)
                        cv2.waitKey(0)

                    for scale in SCALES:
                        aligned_face_im = alignment(cv_im, landmarks_points, scale[0], scale[1])

                        if verbose:
                            cv2.imshow('face_im', aligned_face_im)
                            cv2.waitKey(2)

                        cv2.imwrite('{}/aligned/{}/{}/{}x{}/{}'.format(DATA_ROOT,
                                                                       MSET,
                                                                       SUBJECT,
                                                                       scale[0],
                                                                       scale[1],
                                                                       file),
                                    aligned_face_im)
