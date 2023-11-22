import numpy as np
import sklearn.datasets as skdata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.sequence import pad_sequences


def mean_squared_error(y_hat, y):
    return np.mean((y_hat-y)**2)


data = np.load("data/features.npy", allow_pickle=True)
print(data.shape)
print(np.hstack(data[:, 0:3][0]))
print(data[:, 3].shape)
# flattened_list = [np.hstack(row)[0] for row in data[:, 0:3]]

# x = np.array(flattened_list)
y = data[:, 3]
print("y", np.sum(y == 1))

# print(x.shape)
x = data[:, 0:3]
# x = np.concatenate(x, axis=0).reshape((len(x), -1))
print(x.shape)
print(y.shape)

max_length = max(len(seq) for seq in x.flatten())
print(max_length)
padded_sequences = []

for col_index in range(x.shape[1]):
    # Extract the column
    column = x[:, col_index]
    flattened_column = [arr.flatten().tolist() for arr in column]
    # print("column", column.shape)
    # print("flattened_column", flattened_column)

    # Pad the sequences in the column
    padded_column = pad_sequences(flattened_column, padding='post')

    # Append the padded column to the result
    padded_sequences.append(np.array(padded_column))

# Now, padded_sequences is a list of NumPy arrays with padded sequences

# Stack the padded sequences to get the final result
x_padded = np.column_stack(padded_sequences)
print("padded shape", x_padded.shape)
x_train, x_test, y_train, y_test = train_test_split(x_padded, y, test_size=0.1)

print(x_train.shape)
print(y_train.shape)

model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print(mean_squared_error(predictions, y_test))
print(model.score(x_test, y_test))

logistic_model = LogisticRegression(solver="liblinear")
logistic_model.fit(x_train, y_train)
logistic_pred = logistic_model.predict(x_test)
print(mean_squared_error(logistic_pred, y_test))
print(f"Logistic Regression: {logistic_model.score(x_test,y_test)}")
