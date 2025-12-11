# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import os 


def calc_acc(y, y_pred, classes):
    hits = np.array(y_pred) == np.array(y)
    acc_total = hits.mean()

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # collect the correct predictions for each class
    for label, prediction in zip(y, y_pred):
        if label == prediction:
            correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1

    # print accuracy for each class
    acc_dict = {}
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] > 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            # print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
            acc_dict[classname] = accuracy

    return acc_total, acc_dict


def infer_each_class(net, X, y, classes):
    y_pred = net.predict(X)
    y_proba = net.predict_proba(X)

    acc_total, acc_dict = calc_acc(y, y_pred, classes)

    return y, y_proba, y_pred, (acc_total, acc_dict)


def summarize_results(gs):
    print("Best: %f using %s" % (gs.best_score_, gs.best_params_))
    result = [(mean, stdev, param) for mean, stdev, param in zip(gs.cv_results_['mean_test_score'], gs.cv_results_['std_test_score'], gs.cv_results_['params'])]
    result.sort(key=lambda ele: (ele[0], ele[1]), reverse=True)
    for mean, stdev, param in result:
        print("%f(%f) with: %r" % (mean, stdev, param))

def draw_feature_importance_LogisticRegression(coefficients, feature_names, save_dir=None):
    # Calculate absolute mean of coefficients for multi-class
    if coefficients.ndim > 1:
        mean_abs_coefficients = np.mean(np.abs(coefficients), axis=0)
    else:
        mean_abs_coefficients = np.abs(coefficients)

    # Sort features by importance
    sorted_indices = np.argsort(mean_abs_coefficients)[::-1]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_importances = mean_abs_coefficients[sorted_indices]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_feature_names, sorted_importances)
    plt.xlabel("Feature")
    plt.ylabel("Absolute Mean Coefficient Value")
    plt.title("Logistic Regression Feature Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, 'feature_importance_LR.png')
        plt.savefig(save_file)

def plot_feature_importance(feature_importance, feature_names, save_dir=None):
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.title("Feature Importance")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, 'feature_importance_tree.png')
        plt.savefig(save_file)