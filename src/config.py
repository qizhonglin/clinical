# -*- coding: utf-8 -*-

import os

DATA_ROOT = '/media/qzlin/25793662-6b5a-431d-8402-87c5bd9357df/dataset/LymphNode'
DATA_DIR = os.path.join(DATA_ROOT, 'huaxi/roi')
classes = ('benign', 'malignant')
random_seed=42

CHECKPOINT_DIR = '/media/qzlin/25793662-6b5a-431d-8402-87c5bd9357df/models/haina'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EXTERNAL_DATA_DIR = os.path.join(DATA_ROOT, 'tianfu/roi')