import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def create_features():
    maximum_features = 275
    
    num_llm_essays = None
    with open('data/llm_essays.txt', 'r') as file:
        num_llm_essays = sum(1 for line in file)

    if num_llm_essays is None:
        raise Exception("No essays found")

    llm_features = np.zeros(num_llm_essays, dtype="object") 
    with open('data/llm_essays.txt', 'r') as file:
        index = 0
        for line in file:
            vectorizer = CountVectorizer(max_features=maximum_features)
            essay_features = vectorizer.fit_transform([line.strip()]).toarray()
            padding = np.zeros((1, maximum_features - essay_features.shape[1]))
            essay_features = np.concatenate((essay_features, padding), axis=1)
            # print(essay_features)
            llm_features[index] = np.array([essay_features], dtype="object")
            index += 1
            
    # print(features)

    human_essays = pd.read_csv("data/train_essays.csv")["text"].to_numpy()
    human_features = np.zeros(num_llm_essays, dtype="object") 

    index = 0
    for essay in human_essays:
        if index >= num_llm_essays:
            break
        
        vectorizer = CountVectorizer(max_features=maximum_features)
        essay_features = vectorizer.fit_transform([essay]).toarray()
        padding = np.zeros((1, maximum_features - essay_features.shape[1]))
        essay_features = np.concatenate((essay_features, padding), axis=1)
        human_features[index] = np.array([essay_features], dtype="object")
        index += 1
        
    # print(human_features)

    all_features = np.zeros((num_llm_essays * 2, 2), dtype="object")

    for index, llm_feature in enumerate(llm_features):
        all_features[index,:] = np.array([llm_feature, 1], dtype="object")

    for index, human_feature in enumerate(human_features):
        all_features[index + num_llm_essays,:] = np.array([human_feature, 0], dtype="object")

    # print(all_features)
    # print(all_features.shape)
    np.save("data/all_features_X_y.npy", all_features)
    

create_features()
