from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import *

from sklearn.feature_selection import SelectPercentile, chi2


#from sklearn.cross_validation import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


import numpy as np


def train_model(X_train, y_train, classifier="SVC"):
    clf = None
    if classifier == "SVC":
        clf = SVC()
    elif classifier == "KNN":
        clf = KNeighborsClassifier()
    elif classifier == "GaussianNB":
        clf = GaussianNB()
    elif classifier == "MultinomialNB":
        clf = MultinomialNB()
    elif classifier == "LogisticRegression":
        clf = LogisticRegression()
    elif classifier == "DecisionTreeClassifier":
        clf = DecisionTreeClassifier()
    elif classifier == "RandomForestClassifier":
        clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    return clf

def evaluate(clf, name, X_test, y_test):
    pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    acc = accuracy_score(y_test, pred)
    f1_micro = f1_score(y_test, pred, average='micro')
    f1_macro = f1_score(y_test, pred, average='macro')
    print("Evaluando {}".format(name))
    print("Confusion matrix: ")
    print(cm)
    print("ACC: {}".format(acc))
    print("f1_micro: {}".format(f1_micro))
    print("f1_macro: {}".format(f1_macro))
    print("-------------")

def train_eval(X_train, y_train, X_test, y_test, classifier="SVC"):
    evaluate(train_model(X_train, y_train, classifier=classifier), name=classifier, X_test=X_test, y_test=y_test)