from data import get_data
import numpy as np
import traditional_models
from tfidf import make_tfidf



def show_data():
    X_train, y_train, X_dev, y_dev, X_test = get_data()

    print(len(X_train))
    for x in range(len(X_train)):
        print(X_train[x])
        print("-------------")

if __name__ == "__main__":
    min_ngram, up = 1,7
    min_df = 3
    max_features = None
    X_train, y_train, ids_train, X_dev, y_dev, ids_dev, X_test, id_test = get_data()
    X_train, X_dev, X_test = make_tfidf(X_train, X_dev, X_test, min_ngram,up,max_features=max_features, min_df=min_df)
    clasificadores = ["SVC", "KNN", "GaussianNB", "MultinomialNB", "LogisticRegression", "DecisionTreeClassifier",  "RandomForestClassifier"]
    for clas in clasificadores:
        traditional_models.train_eval(X_train, y_train, X_dev, y_dev, classifier=clas)