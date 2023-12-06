"""
This script is used to train a neural network model using the BERT model.

Plots the training and validation accuracy and loss over the epochs.

Does not return a value.
"""

from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# train_essays = pd.read_csv("data/train_essays.csv")
# train_essays_generated = pd.read_csv("data/train_essays_llm.csv")

train_essays = pd.read_csv("data/train_essays.csv").head(10)
train_essays_generated = pd.read_csv("data/train_essays_llm.csv").head(10)
common_columns = set(train_essays.columns).intersection(
    train_essays_generated.columns)

# Select only the common columns from each DataFrame
df1 = train_essays[common_columns]
df2 = train_essays_generated[common_columns]

# Concatenate the two DataFrames along the rows
combined_df = pd.concat([df1, df2], ignore_index=True)

train_df, test_df = train_test_split(
    combined_df, test_size=0.1, random_state=42)

train_texts = train_df['text'].tolist()
test_texts = test_df['text'].tolist()

train_labels = train_df['generated'].astype(int).tolist()
test_labels = test_df['generated'].astype(int).tolist()

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and encode the training data
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, return_tensors='tf')

# Tokenize and encode the testing data
test_encodings = tokenizer(test_texts, truncation=True,
                           padding=True, return_tensors='tf')

# Unpack the tokenized data
train_input_ids = np.array(train_encodings['input_ids'])
train_attention_mask = np.array(train_encodings['attention_mask'])

# Load the BERT model
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint_callback = ModelCheckpoint(
    'model_weights_{epoch:02d}.h5', save_weights_only=True, period=1)

# Train the model
history = model.fit(
    x=[train_input_ids, train_attention_mask],
    y=np.array(train_labels),  # Convert labels to a NumPy array
    epochs=3,
    callbacks=[checkpoint_callback]
)

test_input_ids = np.array(test_encodings['input_ids'])
test_attention_mask = np.array(test_encodings['attention_mask'])

# Convert test_labels to a NumPy array
test_labels_array = np.array(test_labels)

# Evaluate the model on the testing data
eval_results = model.evaluate(
    [test_input_ids, test_attention_mask], test_labels_array)


# Print the accuracy score
print("Accuracy:", eval_results[1])

plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show(block=True)
