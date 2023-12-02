from models.random_forest import random_forest
from models.svm import svm_model
from models.perceptron import perceptron
from models.naive_bayes import naive_bayes


def main():
    print("========== Testing Models ==========")
    random_forest_accuracy = random_forest()
    print("Random Forest Accuracy:", random_forest_accuracy)
    
    svm_accuracy = svm_model()
    print("SVM Accuracy:", svm_accuracy)
    
    perceptron_accuracy = perceptron()
    print("Perceptron Accuracy:", perceptron_accuracy)
    
    nb_accuracy = naive_bayes()
    print("Naive Bayes Accuracy:", nb_accuracy)
    print("========== Completed ==========")

if __name__ == "__main__":
    main()
    

