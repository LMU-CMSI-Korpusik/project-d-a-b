import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

num_essays = None
with open('data/llm_essays.txt', 'r') as file:
    num_essays = sum(1 for line in file)

with open('data/llm_essays.txt', 'r') as file:
    features = np.zeros((num_essays), dtype="object") if num_essays else Exception("No essays found")
    
    index = 0
    for line in file:
        vectorizer = CountVectorizer()
        essay_features = vectorizer.fit_transform([line.strip()]).toarray()
        features[index] = np.array([essay_features], dtype="object")
        index += 1
        
print(features)