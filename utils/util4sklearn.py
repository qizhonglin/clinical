# -*- coding: utf-8 -*-
import pickle




def infer_each_class(net, X, y, classes):
    y_pred = net.predict(X)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # collect the correct predictions for each class
    for label, prediction in zip(y, y_pred):
        if label == prediction:
            correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    y_proba = net.predict_proba(X)
    return y, y_proba


def summarize_results(gs):
    print("Best: %f using %s" % (gs.best_score_, gs.best_params_))
    result = [(mean, stdev, param) for mean, stdev, param in zip(gs.cv_results_['mean_test_score'], gs.cv_results_['std_test_score'], gs.cv_results_['params'])]
    result.sort(key=lambda ele: (ele[0], ele[1]), reverse=True)
    for mean, stdev, param in result:
        print("%f(%f) with: %r" % (mean, stdev, param))