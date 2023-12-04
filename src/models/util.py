import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def load_and_split_data(test_percentage=0.2):
    data = np.load("data/all_features_X_y.npy", allow_pickle=True)
    X = data[:,0]
    X = np.vstack([item[0] for item in X])
    y = data[:,1]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage)
    return X_train, X_test, y_train, y_test


def score(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


def graph_features():
    X_train, X_test, y_train, y_test = load_and_split_data()
    human_X_train = X_train[y_train == 0]
    llm_X_train = X_train[y_train == 1]
    
    # do PCA on the features
    # plot the first two components
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    human_X_train = pca.fit_transform(human_X_train)
    llm_X_train = pca.fit_transform(llm_X_train)
    
    plt.figure()
    plt.scatter(human_X_train[:,0], human_X_train[:,1], label="Human")
    plt.scatter(llm_X_train[:,0], llm_X_train[:,1], label="LLM")
    plt.legend()
    plt.show(block=True)
    

