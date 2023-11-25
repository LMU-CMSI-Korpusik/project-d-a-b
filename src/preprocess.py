import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def featurize():
    # import train data and prompts as pandas dataframes
    train_essays = pd.read_csv("data/train_essays.csv")
    prompts = pd.read_csv("data/train_prompts.csv")

    # create a dataframe with columns (essay_text, instructions, source_text, generated) with length equal to train_essays length
    corpus = pd.DataFrame(columns=["essay_text", "instructions", "source_text", "generated"], index=np.arange(len(train_essays)))

    for index, row in train_essays.iterrows():
        essay_text = row["text"]
        prompt_id = row["prompt_id"]
        instructions = prompts.loc[prompts["prompt_id"] == prompt_id]["instructions"].values[0]
        source_text = prompts.loc[prompts["prompt_id"] == prompt_id]["source_text"].values[0]
        generated = row["generated"]
        corpus.loc[index] = {"essay_text": essay_text, "instructions": instructions, "source_text": source_text, "generated": generated}

    # encode feature vectors of essay text, instructions, and source text
    # to create vectors < <essay_text_features>, <instruction_features>, <source_text_features>, generated >
    # and put into a numpy array

    features = np.zeros((len(corpus), 4), dtype="object")
    for index, row in corpus.iterrows():
        essay_text = row["essay_text"]
        instructions = row["instructions"]
        source_text = row["source_text"]
        generated = row["generated"]
        essay_text_features = CountVectorizer().fit([essay_text]).transform([essay_text]).toarray()
        instructions_features = CountVectorizer().fit([instructions]).transform([instructions]).toarray()
        source_text_features = CountVectorizer().fit([source_text]).transform([source_text]).toarray()
        features[index,:] = np.array([essay_text_features, instructions_features, source_text_features, generated], dtype="object")

    np.save("data/features.npy", features)


def featurize_essay_text() -> (np.ndarray, np.ndarray):    
    train_essays = pd.read_csv("data/train_essays.csv")
    essay_texts = train_essays["text"]
    generated = train_essays["generated"]

    features = np.zeros((len(train_essays), 1), dtype="object")
    
    for index, essay in enumerate(essay_texts):
        essay_text_features = CountVectorizer().fit([essay]).transform([essay])
        features[index,:] = np.array([essay_text_features], dtype="object")
        
    np.save("data/essay_features.npy", features)
    np.save("data/generated.npy", generated)
    return (features, generated)


def main():
    # featurize()
    # features = np.load("data/features.npy", allow_pickle=True)
    featurize_essay_text()
    

if __name__ == "__main__":
    main()