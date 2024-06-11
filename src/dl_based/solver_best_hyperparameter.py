# -*- coding: utf-8 -*-
"""
split dataset into train (80%), test(20%)
search optimal hyperparameter in config dictionary with val dataset (convered in train dataset) by solver_raytune.py
train model with all train dataset (80%)
test model with model (trained by 80%)
"""

import os
import time
import matplotlib.pyplot as plt
import json

import torch

from src.config import CHECKPOINT_DIR, classes
from src.utils.util import get_logger
from src.utils.util4torch import net2device
from src.utils.SensitivitySpecificityStatistics import SensitivitySpecificityStatistics
from src.dl_based.dataset import get_dataset_train_test, get_dataset_external_test
from src.dl_based.Model2D import Model
from src.dl_based.solver import infer_each_class


def main():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(CHECKPOINT_DIR, f'{timestamp}.log')
    logger = get_logger(name='haina_best', log_file=log_file, log_level='INFO')

    train_ds, test_ds = get_dataset_train_test()
    logger.info(
        f"length of train {len(train_ds)}, length of test {len(test_ds)}")

    config_file = os.path.join(CHECKPOINT_DIR, 'best_raytune.json')
    config = json.loads(open(config_file).read())
    logger.info(f"hyperparameter is {config}")

    model_file = os.path.join(CHECKPOINT_DIR, 'best_hyperparameter.pth')
    # train_config(config, train_ds, None, model_file)

    fix_depth = config["fix_depth"]
    backbone = config["backbone"]
    net = Model(fix_depth=fix_depth).model_define(backbone=backbone, n_class=len(classes))
    net2device(net)
    net.load_state_dict(torch.load(model_file))

    truth, probs = infer_each_class(net, test_ds, classes)
    SensitivitySpecificityStatistics(truth, probs[:, 1], 'internal-test-dl')

    external_test_ds = get_dataset_external_test()
    truth, probs = infer_each_class(net, external_test_ds, classes)
    SensitivitySpecificityStatistics(truth, probs[:, 1], 'external-test-dl')


if __name__ == '__main__':
    main()

    plt.show()