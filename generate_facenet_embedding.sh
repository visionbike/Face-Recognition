#!/usr/bin/env bash
export PYTHONPATH=models/facenet/src

python generate_facenet_embedding.py --model models/facenet/models/20180402-114759 --image_size 160,160 --data_root dataset --mset train
python generate_facenet_embedding.py --model models/facenet/models/20180408-102900 --image_size 160,160 --data_root dataset --mset train