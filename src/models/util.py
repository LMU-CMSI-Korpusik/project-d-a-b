import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_split_data(test_percentage=0.2):
    data = np.load("data/all_features_X_y.npy", allow_pickle=True)
    X = data[:,0]
    X = np.vstack([item[0] for item in X])
    y = data[:,1]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage)
    return X_train, X_test, y_train, y_test