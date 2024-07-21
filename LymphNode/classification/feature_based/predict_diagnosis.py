# -*- coding: utf-8 -*-
import os
import pickle
from pprint import pprint
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from dataset import get_train_test_from_excel, get_external_test_from_excel
from LymphNode.config import CHECKPOINT_DIR, classes
from utils.single_classifier import classify


def main():
    (dfX_train, dfy_train), (dfX_test, dfy_test) = get_train_test_from_excel()
    columns = dfX_train.columns
    (X_train, y_train), (X_test, y_test) = (dfX_train.values, dfy_train.values), (dfX_test.values, dfy_test.values)

    result = classify(X_train, y_train, X_test, y_test, classes,
                                   os.path.join(CHECKPOINT_DIR, f'feature_based'))
    print(result)


if __name__ == '__main__':
    main()

    plt.show()