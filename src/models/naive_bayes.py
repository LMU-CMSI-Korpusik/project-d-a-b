from sklearn.naive_bayes import GaussianNB
from models.util import load_and_split_data
from sklearn.metrics import accuracy_score


def naive_bayes(num_trials = 100):
    X_train, X_test, y_train, y_test = load_and_split_data()

    average_accuracy = 0
    for _ in range(num_trials):
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        average_accuracy += accuracy_score(y_test, y_pred)

    return average_accuracy / num_trials