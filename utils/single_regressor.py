# -*- coding: utf-8 -*-
"""
:Authors: Qizhong Lin <qizhong.lin@philips.com>,
:Copyright: This file contains proprietary information of Philips 
            Innovative Technologies. Copying or reproduction without prior
            written approval is prohibited.

            Philips internal use only - no distribution outside Philips allowed
"""
import os
from sklearn.pipeline import Pipeline
from sklearn import linear_model, neural_network, ensemble, svm, naive_bayes, neighbors, tree
from xgboost import XGBRegressor
from skopt import BayesSearchCV
import pandas as pd
import pickle

random_state = 42


def regression(X_train, y_train, X_test, y_test, checkpoint_dir, verbose=False):
    """
    refer to single_classifier/classify(..)

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param checkpoint_dir:
    :param verbose:
    :return:
    """
    feature = os.path.split(checkpoint_dir)[-1]

    model_file = os.path.join(checkpoint_dir, "model.pkl")
    if not os.path.exists(model_file):
        pipe = Pipeline([('Regressor', ensemble.RandomForestRegressor(random_state=random_state))])

        params = [
            # {
            #     'Regressor': [ensemble.RandomForestRegressor(random_state=random_state)],
            #     'Regressor__criterion': ['friedman_mse'],
            #     'Regressor__n_estimators': [50, 100, 500],
            # },
            # {
            #     'Regressor': [tree.DecisionTreeRegressor(random_state=random_state)],
            #     'Regressor__criterion': ['friedman_mse'],
            #     'Regressor__max_depth': [None, 1, 5, 10, 15],
            # },
            # {
            #     'Regressor': [XGBRegressor(random_state=random_state)],
            #     'Regressor__learning_rate': [0.01, 0.1, 0.2, 0.5],
            # },
            # {
            #     'Regressor': [ensemble.LGBMRegressor(random_state=random_state)],
            #     'Regressor__num_leaves': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            #     'Regressor__learning_rate': [0.01, 0.1, 0.2, 0.5],
            # },
            # {
            #     'Regressor': [ensemble.CatBoostRegressor(verbose=False, random_state=random_state)],
            # },
            # {
            #     'Regressor': [linear_model.ElasticNet()],
            # },
            # {
            #     'Regressor': [neural_network.MLPRegressor(max_iter=500)],
            # },
            # {
            #     'Regressor': [svm.SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)],
            # },
            {
                'Regressor': [ensemble.GradientBoostingRegressor()],
                "Regressor__n_estimators": [500],
                "Regressor__max_depth": [4],
                "Regressor__min_samples_split": [5],
                "Regressor__learning_rate": [0.01],
                "Regressor__loss": ["squared_error"]
            }
        ]

        optimize = BayesSearchCV(pipe, params, scoring='neg_mean_absolute_error', n_iter=50*len(params), cv=3, n_jobs=-1)
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

    truths = y_test
    preds = model.predict(X_test)

    return truths, preds
