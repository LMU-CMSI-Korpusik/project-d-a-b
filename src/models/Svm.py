import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



def main():

    features = np.load("C:\\Users\\1149896\\Documents\\cmsi-5350\\project-d-a-b\\data\\essay_features.npy", allow_pickle=True)
    y = np.load("C:\\Users\\1149896\\Documents\\cmsi-5350\\project-d-a-b\\data\\generated.npy", allow_pickle=True)
  

    # print(y)
    print(features)
    


if __name__ == "__main__":
    main()