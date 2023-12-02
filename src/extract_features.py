import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# load in training data
print("Loading training data...")
llm_essays = pd.read_csv("data/train_essays_llm_555.csv")
num_llm_essays = len(llm_essays)

human_essays_all_prompts = pd.read_csv("data/train_essays.csv")
human_essays = human_essays_all_prompts.loc[human_essays_all_prompts["prompt_id"] == 0]["text"].to_numpy()
human_essays = np.random.choice(human_essays, num_llm_essays, replace=False)

# create features
print("Creating features...")
maximum_features = 275
llm_features = np.zeros(num_llm_essays, dtype="object")
human_features = np.zeros(num_llm_essays, dtype="object")

print("LLM features...")
for index, essay in enumerate(llm_essays["text"]):
    vectorizer = CountVectorizer(max_features=maximum_features)
    essay_features = vectorizer.fit_transform([essay]).toarray()
    padding = np.zeros((1, maximum_features - essay_features.shape[1]))
    essay_features = np.concatenate((essay_features, padding), axis=1)
    llm_features[index] = np.array([essay_features], dtype="object")

print("Human features...")
for index, essay in enumerate(human_essays):
    vectorizer = CountVectorizer(max_features=maximum_features)
    essay_features = vectorizer.fit_transform([essay]).toarray()
    padding = np.zeros((1, maximum_features - essay_features.shape[1]))
    essay_features = np.concatenate((essay_features, padding), axis=1)
    human_features[index] = np.array([essay_features], dtype="object")

# combine features and labels into one array
print("Combining features and labels...")
all_features = np.zeros((num_llm_essays * 2, 2), dtype="object")

for index, llm_feature in enumerate(llm_features):
    all_features[index,:] = np.array([llm_feature, 1], dtype="object")
    
for index, human_feature in enumerate(human_features):
    all_features[index + num_llm_essays,:] = np.array([human_feature, 0], dtype="object")

# save features and labels
np.save("data/all_features_X_y.npy", all_features)

print("Done!")