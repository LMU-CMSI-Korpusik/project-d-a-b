from models.random_forest import random_forest
from models.svm import svm_model
from models.perceptron import perceptron
from models.naive_bayes import naive_bayes


def main():
    print("========== Testing Models ==========")
    print("Running 100 trials for each model")
    print("Accuracy, Precision, Recall, F1")
    print("====================================")
    random_scores = random_forest()
    print("Random Forest Scores:", random_scores)
    
    svm_scores = svm_model()
    print("SVM Scores:", svm_scores)
    
    perceptron_scores = perceptron()
    print("Perceptron Scores:", perceptron_scores)
    
    nb_scores = naive_bayes()
    print("Naive Bayes Scores:", nb_scores)
    print("========== Completed ==========")

if __name__ == "__main__":
    main()
    

