"""
Exports the embeddings and labels of a directory of images as numpy arrays.

Typicall usage expect the image directory to be of the openface/facenet form and
the images to be aligned. Simply point to your model and your image directory:
    python facenet/contributed/export_embeddings.py ~/models/facenet/20170216-091149/ ~/datasets/lfw/mylfw

Output:
embeddings.npy -- Embeddings as np array, Use --embeddings_name to change name
labels.npy -- Integer labels as np array, Use --labels_name to change name
label_strings.npy -- Strings from folders names, --labels_strings_name to change name


Use --image_batch to dictacte how many images to load in memory at a time.

If your images aren't already pre-aligned, use --is_aligned False

I started with compare.py from David Sandberg, and modified it to export
the embeddings. The image loading is done use the facenet library if the image
is pre-aligned. If the image isn't pre-aligned, I use the compare.py function.
I've found working with the embeddings useful for classifications models.

Charles Jekel 2017

"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import argparse
import facenet
import align.detect_face
import cv2
from tqdm import *
from imgaug import augmenters as iaa
from helper import make_if_not_exist

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--model', type=str, default='', help='path to lad model.')
parser.add_argument('--image_size', type=str, default='160,160', help='image size (height, width) in pixels')
parser.add_argument('--data_root', type=str, default='', help='directory to store face images')
parser.add_argument('--mset', type=str, default='train', help='train/test mset')
parser.add_argument('--ignore_unknown', default=False, action='store_true', help='include unknown class or not')
args = parser.parse_args()

sometimes = lambda aug: iaa.Sometimes(0.8, aug)
seq = iaa.Sequential([iaa.Fliplr(0.5),
                      sometimes(iaa.OneOf([iaa.Grayscale(alpha=(0.0, 1.0)),
                                           iaa.AddToHueAndSaturation((-20, 20)),
                                           iaa.Add((-20, 20), per_channel=0.5),
                                           iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                           iaa.GaussianBlur((0, 2.0)),
                                           iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                                           iaa.Sharpen(alpha=(0, 0.5), lightness=(0.7, 1.3)),
                                           iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.5))])
                                )
                      ])


def pre_whitten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


DATA_ROOT = 'dataset'

if __name__ == '__main__':
    im_width, im_height = args.image_size.split(',')
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(args.model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            output_dir = 'models/facenet/embeddings/{}/{}'.format(args.model.split('/')[-1], args.mset)
            make_if_not_exist(output_dir)

            for subject in os.listdir('{}/aligned/{}'.format(args.data_root, args.mset)):
                if subject == 'unknown' and args.ignore_unknown:
                    continue
                for r, _, files in os.walk('{}/aligned/{}/{}/{}x{}'.format(args.data_root, args.mset, subject, im_width, im_height)):
                    for file in tqdm(files):
                        if '.png' not in file:
                            continue
                        fn, fe = os.path.splitext(file)
                        im_path = os.path.join(r, file)
                        im_org = cv2.imread(im_path)
                        im_org - cv2.cvtColor(im_org, cv2.COLOR_BGR2RGB)

                        im = pre_whitten(im_org)
                        im = np.expand_dims(im, axis=0)
                        feed_dict = {images_placeholder: im, phase_train_placeholder: False}
                        embedding = sess.run(embeddings, feed_dict=feed_dict)
                        np.save(output_dir + '/{}_{}.npy'.format(subject, fn), embedding)

                        if args.mset == 'test':
                            im_flip = cv2.flip(im_org, 1)
                            im_flip = pre_whitten(im_flip)
                            im_flip = np.expand_dims(im_flip, axis=0)
                            feed_dict = {images_placeholder: im_flip, phase_train_placeholder: False}
                            embeding = sess.run(embeddings, feed_dict=feed_dict)
                            np.save(output_dir + '/{}_{}_flip.npy'.format(subject, fn), embeding)

                        augmentation_arr = np.array([], dtype=np.float32).reshape(0, 512)
                        for i in range(100):
                            im_aug = seq.augment_image(im_org)
                            im_aug = pre_whitten(im_aug)
                            im_aug = np.expand_dims(im_aug, axis=0)
                            feed_dict = {images_placeholder: im_aug, phase_train_placeholder: False}
                            embedding = sess.run(embeddings, feed_dict=feed_dict)
                            augmentation_arr = np.vstack((augmentation_arr, embedding.reshape(1, 512)))
                        np.save(output_dir + '/{}_{}_augmentation.npy'.format(subject, fn), augmentation_arr)
