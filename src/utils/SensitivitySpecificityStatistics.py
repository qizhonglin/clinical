#!/usr/bin/env python3

# Copyright (c) 2017-present, Philips, Inc.

"""
-------------------------------------------------
   File Name：     SensitivitySpecificityStatistics
   Description :
   Author :        qizhong.lin@philips.coom
   date：          21-1-10
-------------------------------------------------
   Change Activity:
                   21-1-10:
-------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, auc, roc_curve


class SensitivitySpecificityStatistics(object):
    def __init__(self, y_test, y_pred_score, model_name):
        self.y_test = y_test
        self.y_pred_score = y_pred_score

        self.FPR, self.TPR, self.scores = roc_curve(self.y_test, self.y_pred_score)
        # self.calc_sen_spe()

        self.AUC = self.plot_roc(title=model_name)

    def calc_sen_spe(self, num=200):
        y_test = self.y_test
        y_pred_score = self.y_pred_score

        recall = np.zeros((num,))
        sensitivity = np.zeros((num,))
        specificity = np.zeros((num,))
        scores = np.zeros((num,))
        for idx, thresh in enumerate(np.linspace(0, 1, num=num)):
            y_pred = y_pred_score >= thresh
            true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, y_pred).ravel()
            recall[idx] = recall_score(y_test, y_pred, average='macro')
            sensitivity[idx] = true_positive / (true_positive + false_negative)
            specificity[idx] = true_negative / (true_negative + false_positive)
            scores[idx] = thresh

        self.TPR = recall[::-1]        # from small to large
        self.FPR = 1 - specificity[::-1]    # from small to large, specificity from large to small
        self.scores = scores[::-1]

    def plot_roc(self, title):
        FPR, TPR, scores = self.FPR, self.TPR, self.scores

        plt.figure()
        plt.plot(FPR, TPR)
        plt.fill_between(FPR, TPR, alpha=0.2, color='b')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.xlabel('False Positive Rate (1-specitifity)')
        plt.ylabel('True Positive Rate (sensitivity')

        AUC = auc(FPR, TPR)
        plt.title('{0}--TPR-FPR Curve: AUC={1:0.4f}\n(Sensitivity, Specificity, Score)'.format(title, AUC))

        idx = self.optimal_thresh_by_sensitivity(thresh=0.35)
        self.add_thresh(FPR[idx], TPR[idx], scores[idx], color='b')
        idx = self.optimal_thresh_by_sensitivity(thresh=0.6)
        self.add_thresh(FPR[idx], TPR[idx], scores[idx], color='b')
        # idx = self.optimal_thresh_by_sensitivity(thresh=0.77)
        # self.add_thresh(FPR[idx], TPR[idx], scores[idx], color='b')
        idx = self.optimal_thresh_by_sensitivity(thresh=0.83)
        self.add_thresh(FPR[idx], TPR[idx], scores[idx], color='b')
        # idx = self.optimal_thresh_by_thresh()
        # self.add_thresh(FPR[idx], TPR[idx], scores[idx], color='b')

        return AUC

    def add_thresh(self, FPR, TPR, thresh, color='r', ymax=1, xmax=1, text_delta=0):
        plt.axvline(x=FPR, ymax=TPR/ymax, ls='--', color=color)
        plt.axhline(y=TPR, xmax=FPR/xmax, ls='--', color=color)
        plt.scatter([FPR, ], [TPR, ], s=20, color=color)
        plt.text(FPR, TPR+text_delta, '({0:0.2f}, {1:0.3f}, t={2:0.3f})'.format(TPR, 1-FPR, thresh),
                 fontdict={'size': 10, 'color': color})

    def optimal_thresh_by_sensitivity(self, thresh=0.6):
        for i, value in enumerate(self.TPR):
            if value >= thresh:
                return i
        return 0

    def optimal_thresh_by_thresh(self, thresh=0.5):
        for i, score in enumerate(self.scores):
            if score < thresh:
                return max(i-1, 0)
        return 0


def plot_roc(y_test, y_pred_score, y_test1, y_pred_score1):
    FPR, TPR, scores = roc_curve(y_test, y_pred_score)
    FPR1, TPR1, scores1 = roc_curve(y_test1, y_pred_score1)

    # plt.figure()
    plt.plot(FPR, TPR)
    plt.plot(FPR1, TPR1)
    # plt.fill_between(FPR, TPR, alpha=0.2, color='b')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('False Positive Rate (1-specitifity)')
    plt.ylabel('True Positive Rate (sensitivity')

    # AUC = auc(FPR, TPR)
    # plt.title('{0}--TPR-FPR Curve: AUC={1:0.4f}\n(Sensitivity, Specificity, Score)'.format(title, AUC))