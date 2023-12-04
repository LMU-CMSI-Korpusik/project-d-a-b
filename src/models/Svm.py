import pandas as pd
import numpy as np
from sklearn import svm
from models.util import load_and_split_data, score

def svm_model(num_trials = 100):
    X_train, X_test, y_train, y_test = load_and_split_data()

    average_accuracy = 0
    average_precision = 0
    average_recall = 0
    average_f1 = 0
    for _ in range(num_trials):
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy, precision, recall, f1 = score(y_test, y_pred)
        average_accuracy += accuracy
        average_precision += precision
        average_recall += recall
        average_f1 += f1

    return average_accuracy / num_trials, average_precision / num_trials, average_recall / num_trials, average_f1 / num_trials