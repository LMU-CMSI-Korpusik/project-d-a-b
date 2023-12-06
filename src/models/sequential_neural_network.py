"""
This script is used to train a sequential neural network model.

Displays the classification report and confusion matrix of the model.

Does not return a value.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

train_essays = pd.read_csv("data/train_essays.csv").head(555)
train_essays_generated = pd.read_csv("data/train_essays_llm_555.csv")

common_columns = set(train_essays.columns).intersection(
    train_essays_generated.columns)

df1 = train_essays[common_columns]
df2 = train_essays_generated[common_columns]

combined_df = pd.concat([df1, df2], ignore_index=True)
data_texts = combined_df['text'].tolist()
data_labels = combined_df['generated'].astype(int).tolist()

vocab_size = 20000
embedding_dim = 100
max_length = 512

tokenizer_seq = Tokenizer(oov_token="<OOV>")
tokenizer_seq.fit_on_texts(data_texts)

train_sequences = tokenizer_seq.texts_to_sequences(data_texts)

train_padded = pad_sequences(
    train_sequences, maxlen=max_length, padding='post', truncating='post')

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(data_labels)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim,
              input_length=max_length),
    LSTM(128),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_padded, val_padded, train_labels_encoded, val_labels_encoded = train_test_split(
    train_padded, train_labels_encoded, test_size=0.2, random_state=42
)

history = model.fit(
    x=train_padded,
    y=train_labels_encoded,
    epochs=4,
    validation_data=(val_padded, val_labels_encoded)
)

eval_results = model.evaluate(val_padded, val_labels_encoded)
print("Test Accuracy:", eval_results[1])

predictions = model.predict(val_padded)
predicted_labels = np.argmax(predictions, axis=1)

predicted_labels_original = label_encoder.inverse_transform(predicted_labels)
test_labels_original = label_encoder.inverse_transform(val_labels_encoded)

print("Classification Report:")
print(classification_report(test_labels_original, predicted_labels_original))

cm = confusion_matrix(test_labels_original, predicted_labels_original)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show(block=True)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show(block=True)
