import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



data = np.load("C:\\Users\\1149896\\Documents\\cmsi-5350\\project-d-a-b\\data\\all_features_X_y.npy", allow_pickle=True)
X = data[:,0]
X = np.vstack([item[0] for item in X])
y = data[:,1]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# varying alpha from .0001 to .1
alpha = np.linspace(.0001,.1, 100)
train_score = []
test_score = []
train_array = []
test_array = []

# iterate through each alpha value from .0001 to .1
for i in alpha:

# 100 different runs per value of alpha and create an array of scores
    for run in range(100):
        model = Perceptron(penalty = 'l1', alpha = i)
        model.fit(X_train,y_train)
        RunScore_train = model.score(X_train, y_train) 
        RunScore_test = model.score(X_test, y_test)
        train_score = np.append(train_score, RunScore_train)
        test_score = np.append(test_score,RunScore_test)  

# Take the average test/train score over the course of 100 different runs per alpha value and create an array for each value of alpha
    train_mean = np.mean(train_score)
    test_mean = np.mean(test_score)
    train_array = np.append(train_array, train_mean)
    test_array = np.append(test_array, test_mean) 

#Plot each array
plt.plot(alpha, train_array, label = 'Train Score')
plt.plot(alpha, test_array, label = 'Test Score')
plt.xlabel('Alpha Value')
plt.ylabel('Test Accuracy')
plt.legend()
plt.show()

