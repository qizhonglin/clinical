# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from dataset import get_train_test_from_excel, get_external_test_from_excel, map_chinese2english
from LymphNode.config import CHECKPOINT_DIR, classes
from utils.single_classifier import classify


def main():
    # config = {
    #     'is_tree': False,
    #     'outputs': 'outputs',
    # }
    config = {
        'is_tree': True,
        'outputs': 'outputs_tree',
    } 
    has_preprocess = False if config['is_tree'] else True
        
    (dfX_train, dfy_train), (dfX_test, dfy_test) = get_train_test_from_excel(has_preprocess=has_preprocess)
    columns = dfX_train.columns
    columns_en = map_chinese2english(dfX_train.columns)
    (X_train, y_train), (X_test, y_test) = (dfX_train.values, dfy_train.values), (dfX_test.values, dfy_test.values)

    dfX_test_ext, dfy_test_ext = get_external_test_from_excel(has_preprocess=has_preprocess)
    X_test_ext, y_test_ext = dfX_test_ext.values, dfy_test_ext.values

    data_info = {
        'train': len(X_train),
        'internal test': len(X_test),
        'external test': len(X_test_ext)
    }
    print(data_info)
    exit()

    config['classes'] = classes
    config['feature_names'] = columns_en

    result = classify(X_train, y_train, X_test, y_test, config,
                      os.path.join(CHECKPOINT_DIR, f'feature_based', config['outputs']),
                      X_test_ext, y_test_ext, verbose=True)
    print(result)



if __name__ == '__main__':
    main()

    plt.show()
