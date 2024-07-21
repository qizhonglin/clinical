# -*- coding: utf-8 -*-
import os
import pickle
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn import linear_model, neural_network, ensemble, svm, naive_bayes, neighbors, tree
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from utils.SensitivitySpecificityStatistics import SensitivitySpecificityStatistics
from utils.util4sklearn import infer_each_class


random_state = 42


def classify(X_train, y_train, X_test, y_test, classes, checkpoint_dir, verbose=False):
    """

    :param X_train: (sample num, feature num)
    :param y_train: (sample num, ) , could be binary or multi-class
    :param X_test:
    :param y_test:
    :param classes: (class1, class2, class3, ...)
    :param checkpoint_dir: the directory to save model
    :param verbose: if True, print the process information
    :return: {
        'accuracy': acc_total,
        'accuracy detail': acc_dict,
        'AUC':                  (only exist for binary class)
    }
    """
    feature = os.path.split(checkpoint_dir)[-1]
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    if verbose: print_distribution(y_train, y_test, classes)

    model_file = os.path.join(checkpoint_dir, "model.pkl")
    if not os.path.exists(model_file):
        pipe = Pipeline([('Classifier', linear_model.LogisticRegression(random_state=random_state))])
        params = [
            {
                'Classifier': [linear_model.LogisticRegression(C=1.0, penalty='l1', solver='saga')],
                'Classifier__C': np.logspace(1, 3, 3),
                'Classifier__penalty': ["l1", "l2", "elasticnet"],
                'Classifier__l1_ratio': np.linspace(0, 1, 5),
                'Classifier__class_weight':  ['balanced', None],
            },
            {
                'Classifier': [ensemble.AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(max_depth=2, max_leaf_nodes=8), algorithm="SAMME",)],
                'Classifier__n_estimators': [100, 200, 300, 500],
            },
            {
                'Classifier': [ensemble.ExtraTreesClassifier()],
                'Classifier__n_estimators': [100, 200, 300, 500],
                'Classifier__max_features': ['log2', 'sqrt'],
                'Classifier__min_samples_split': [2, 3, 5],
            },
            {
                'Classifier': [ensemble.RandomForestClassifier()],
                'Classifier__n_estimators': [100, 200, 300, 500],
                'Classifier__max_features': ['log2', 'sqrt'],
                'Classifier__max_depth': [1, 2],
            },
            {
                'Classifier': [XGBClassifier(reg_alpha=3, reg_lambda=3)],
                'Classifier__n_estimators': [100, 200, 300, 500],
                'Classifier__max_depth': [4, 5, 6],
                'Classifier__min_child_weight':  [1, 3, 5],
                'Classifier__gamma': [0, 0.5, 1]
            },
            {
                'Classifier': [neural_network.MLPClassifier()],
                'Classifier__hidden_layer_sizes': [100, 200, 300],
                'Classifier__max_iter': [300],
            },
            {
                'Classifier': [svm.SVC(gamma=0.001, C=0.0001, probability=True)],
                'Classifier__C': Real(1e-6, 1e+6, prior='log-uniform'),
                'Classifier__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                'Classifier__degree': Integer(1, 8),
                'Classifier__kernel': Categorical(['linear', 'poly', 'rbf']),
                'Classifier__class_weight': ['balanced', None]
            },
            {
                'Classifier': [neighbors.KNeighborsClassifier()],
                'Classifier__n_neighbors': [i for i in range(2, 4)],
                'Classifier__weights': ['uniform', 'distance'],
            },
        ]
        # optimize = GridSearchCV(pipe, params, cv=3, n_jobs=-1)
        optimize = BayesSearchCV(pipe, params, n_iter=50*len(params), cv=3, n_jobs=-1)
        optimize.fit(X_train, y_train)

        if verbose:
            cv_results_df = pd.DataFrame(optimize.cv_results_)
            print(cv_results_df)
            print(f"best params: {optimize.best_params_}")

        model = optimize.best_estimator_

        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

    # use trained model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    truth, probs, preds, (acc_total, acc_dict) = infer_each_class(model, X_test, y_test, classes)

    info = {
        'accuracy': acc_total,
        'accuracy detail': acc_dict
    }
    if len(classes) == 2:
        AUC_internal = SensitivitySpecificityStatistics(y_test, probs[:, 1], feature).AUC
        info['AUC'] = AUC_internal

    return info


def print_distribution(y_train, y_test, classes):
    class_distribution = {'train': [], 'test': []}
    for i, clc in enumerate(classes):
        ratio_train = sum(y_train == i) / len(y_train)
        ratio_test = sum(y_test == i) / len(y_test)

        class_distribution['train'].append(ratio_train)
        class_distribution['test'].append(ratio_test)
    print(f"class distribution: {class_distribution}")