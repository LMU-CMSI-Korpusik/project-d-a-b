import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

data = np.load("data/all_features_X_y.npy", allow_pickle=True)
X = data[:,0]
X = np.vstack([item[0] for item in X])
y = data[:,1]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

average_accuracy = 0
for i in range(100):
    clf = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    average_accuracy += accuracy_score(y_test, y_pred)

print(average_accuracy / 100)
