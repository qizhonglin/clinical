# -*- coding: utf-8 -*-
import os
import pickle
from pprint import pprint
from sklearn import svm, ensemble, linear_model, neural_network, neighbors
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from dataset import get_train_test_from_excel, get_external_test_from_excel
from src.config import CHECKPOINT_DIR, classes
from src.utils.SensitivitySpecificityStatistics import SensitivitySpecificityStatistics
from src.utils.util4sklearn import summarize_results, infer_each_class


def search_optimal(model, space, X_train, y_train, optimize_mode="GridSearchCV"):
    """

    :param model:
    :param space:
    :param X_train:
    :param y_train:
    :param optimize_mode: one of ["GridSearchCV", "BayesSearchCV]
    :return:
    """
    print('optimize hyper-parameter via grid search')
    optimize = GridSearchCV(model, space, scoring='roc_auc', cv=3, n_jobs=-1)
    if optimize_mode == "BayesSearchCV":
        optimize = BayesSearchCV(model, space, scoring='roc_auc', cv=3, n_jobs=-1)

    optimize.fit(X_train, y_train)
    print(optimize.best_params_)

    return optimize.best_params_


def build_models():
    models = {}

    model = linear_model.LogisticRegression(
        C=1.0, penalty='l1', solver='saga', multi_class='multinomial', class_weight='balanced')
    space = [
        {'C': np.logspace(1, 3, 3)}
    ]
    models["linear_model.LogisticRegression"] = (model, space)

    # model = ensemble.RandomForestClassifier(class_weight='balanced')
    # space = [
    #     {'n_estimators': [10, 100, 500, 1000], 'max_features': ['auto', 'sqrt'], },
    # ]
    # models["ensemble.RandomForestClassifier"] = (model, space)
    #
    # model = XGBClassifier(class_weight='balanced', reg_alpha=3, reg_lambda=3)
    # space = [
    #     {'n_estimators': [100, 500, 1000], 'max_depth': [4, 5, 6], 'min_child_weight': [1, 3, 5],
    #      'gamma': [0, 0.5, 1]}
    # ]
    # models["XGBClassifier"] = (model, space)
    #
    # model = neural_network.MLPClassifier()
    # space = [
    #     {'hidden_layer_sizes': [100, 500, 1000], 'alpha': [10.0 ** i for i in range(-5, 1)]}
    # ]
    # models["neural_network.MLPClassifier"] = (model, space)
    #
    # model = neighbors.KNeighborsClassifier()
    # space = [
    #     {'n_neighbors': [i for i in range(1, 10)], 'weights': ['uniform', 'distance']}
    # ]
    # models["neighbors.KNeighborsClassifier"] = (model, space)
    #
    # model = svm.SVC(gamma=0.001, C=0.0001, probability=True, class_weight='balanced')
    # space = [
    #     {'gamma': np.logspace(-4, -2, 3), 'C': np.logspace(5, 7, 3)}
    # ]
    # models["svm.SVC"] = (model, space)

    return models


def main():
    (dfX_train, dfy_train), (dfX_test, dfy_test) = get_train_test_from_excel()
    columns = dfX_train.columns
    (X_train, y_train), (X_test, y_test) = (dfX_train.values, dfy_train.values), (dfX_test.values, dfy_test.values)

    models = build_models()

    info = {model_name: model.get_params() for model_name, (model, space) in models.items()}
    pprint(info)

    result = []
    for model_name, (model, space) in models.items():
        print(f"train model {model_name}...")
        model_file = os.path.join(CHECKPOINT_DIR, f"feature_bayes.{model_name}.pkl")

        # train model
        # # best_params = search_optimal(model, space, X_train, y_train, optimize_mode="GridSearchCV")
        # best_params = search_optimal(model, space, X_train, y_train, optimize_mode="BayesSearchCV")
        #
        # model.set_params(**best_params)
        # model.fit(X_train, y_train)
        #
        # with open(model_file, 'wb') as f:
        #     pickle.dump(model, f)

        # use trained model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        truth, probs = infer_each_class(model, X_test, y_test, classes)
        AUC_internal = SensitivitySpecificityStatistics(truth, probs[:, 1], f'internal-test-{model_name}').AUC

        X_test_external, y_test_external = get_external_test_from_excel()
        truth, probs = infer_each_class(model, X_test_external, y_test_external, classes)
        AUC_external = SensitivitySpecificityStatistics(truth, probs[:, 1], f'external-test-{model_name}').AUC

        result.append((model_name, AUC_internal, AUC_external))

    result.sort(key=lambda ele: (ele[1], ele[2]))
    pprint(result)


if __name__ == '__main__':
    main()

    plt.show()