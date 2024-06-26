# Using AI to Detect AI:
## The Efficacy of Various ML Models for Detecting LLM Generated Text

Authors: Abe Moore Odell, Brady Amundson, Denis Onwualu

### Overview
Large language model (llm) generated text has the potential for being used for malicious purposes, including spam, phishing, and the creation of disinformation. This project seeks to leverage ML models to identify if a given text was (0) written by a human, or (1) generated by a LLM, specifically chat-gpt 3.5. Initial results prove promising with high model accuracies compared to the 59.38 % accuracy of polled human deciders.

### Running the Program
1. If all_features_X_y.npy is not present in the data directory, or if you would like to update the features, run src/preprocessing/extract_features.py to generate features.
2. Run src/main.py to test the base models (random_forest, SVM, perceptron, naive_bayes).
3. In order to run, test, or modify a specific model, navigate to src/models.