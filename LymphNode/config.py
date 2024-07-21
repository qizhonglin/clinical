# -*- coding: utf-8 -*-

import os
random_seed = 42

DATA_ROOT = '/media/qzlin/25793662-6b5a-431d-8402-87c5bd9357df/dataset/LymphNode'
DATA_DIR = os.path.join(DATA_ROOT, 'huaxi/roi')
classes = ('benign', 'malignant')
CHECKPOINT_DIR = os.path.join(DATA_ROOT, 'model')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EXTERNAL_DATA_DIR = os.path.join(DATA_ROOT, 'tianfu/roi')
is_train_with_external_data = False
RAYTUNE = 'raytune_tunebohb_train_with_external_data' if is_train_with_external_data else 'raytune_tunebohb_train_without_external_data'