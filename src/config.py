# -*- coding: utf-8 -*-

import os
random_seed=42

DATA_ROOT = '/media/qzlin/25793662-6b5a-431d-8402-87c5bd9357df/dataset/LymphNode'
DATA_DIR = os.path.join(DATA_ROOT, 'huaxi/roi')
EXTERNAL_DATA_DIR = os.path.join(DATA_ROOT, 'tianfu/roi')
classes = ('benign', 'malignant')
CHECKPOINT_DIR = os.path.join(DATA_ROOT, 'model')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
RAYTUNE = 'raytune_tunebohb'

