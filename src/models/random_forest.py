import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from util import load_and_split_data


X_train, X_test, y_train, y_test = load_and_split_data()

average_accuracy = 0
for i in range(100):
    clf = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    average_accuracy += accuracy_score(y_test, y_pred)

print(average_accuracy / 100)
