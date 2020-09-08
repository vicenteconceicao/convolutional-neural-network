import sys
import os
import csv
import time

from sklearn.linear_model import Perceptron

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_svmlight_file

# Cria diretório se não existir.
def createDirectory(path):
        if not os.path.isdir(path):
                os.mkdir(path)


if __name__ == "__main__":

    model = "xception"

    # loads data.
    features_path = "./features/"+model+"_train.svm"
    print ("Loading train data...")
    X_train, y_train = load_svmlight_file(features_path)
    X_train = X_train.toarray()

    features_path = "./features/"+model+"_test.svm"
    print ("Loading test data...")
    X_test, y_test = load_svmlight_file(features_path)
    X_test = X_test.toarray()

    # Fitting Perceptron.
    print("Fitting Perceptron")
    clf = Perceptron(tol=1e-3, random_state=0)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    fit_time = (end - start)

    # Predicting Perceptron.
    print("Preditting Perceptron")
    start = time.time()
    y_pred = clf.predict(X_test)
    end = time.time()
    predict_time = (end - start)

    createDirectory("./result") 
    result_file = open("./result/"+model+".txt", "a+") 

    #Accurary
    accuracy = str(clf.score(X_test, y_test))
    result_file.write("accuracy:"+accuracy+"\n")

    #F1 Score
    f1s = str(f1_score(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11], average='weighted'))
    result_file.write("f1s:"+f1s+"\n")

    #Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    result_file.write(str(cm))

    result_file.close()