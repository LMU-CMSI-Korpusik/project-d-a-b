import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from models.util import load_and_split_data

def random_forest(num_trials = 100):
    X_train, X_test, y_train, y_test = load_and_split_data()

    average_accuracy = 0
    for _ in range(num_trials):
        clf = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        average_accuracy += accuracy_score(y_test, y_pred)

    return average_accuracy / num_trials
