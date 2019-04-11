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

from sklearn.model_selection import GridSearchCV
import numpy as np


def train_model(X_train, y_train, classifier="SVC"):
    clf = None
    if classifier == "SVC":
        tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},]
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='f1_macro')
    elif classifier == "KNN":
        tuned_parameters = [{'weights': ['uniform', 'distance'], 'n_neighbors': [3,4,5,6],
                             'algorithm': ['auto', 'ball_tree', 'kd_tree']}, ]
        clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='f1_macro')
    elif classifier == "GaussianNB":
        tuned_parameters = [{'var_smoothing': [1e-9, 1e-8]}, ]
        clf = GridSearchCV(GaussianNB(), tuned_parameters, cv=5, scoring='f1_macro')
    elif classifier == "MultinomialNB":
        tuned_parameters = [{'alpha': [0,1.0,0.5,1.5], 'fit_prior':[True,False]}, ]
        clf = GridSearchCV(MultinomialNB(), tuned_parameters, cv=5, scoring='f1_macro')
    elif classifier == "LogisticRegression":
        tuned_parameters = [{'penalty': ['l1', 'l2'], 'C': [1.0,0.5,2.0,10.0]}, ]
        clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5, scoring='f1_macro')
    elif classifier == "DecisionTreeClassifier":
        clf = DecisionTreeClassifier()
    elif classifier == "RandomForestClassifier":
        clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))

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