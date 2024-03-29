from sklearn.feature_extraction.text import *
from nltk.tokenize import TweetTokenizer
from parser import get_tweets
import numpy as np

def get_data(token=False):
    X_train_, y_train = [], []
    """Parser"""
    path = "data/"
    training_path = path + 'TASS2017_T1_training.xml'
    dev_path = path + 'TASS2017_T1_development.xml'
    test_path = path + 'TASS2017_T1_test.xml'
    X_train_, y_train, ids_train = get_tweets(training_path)
    X_dev, y_dev, ids_dev = get_tweets(dev_path)
    X_test, ids_test = get_tweets(test_path, test=True)

    print("Datos train: {} \nDatos Dev: {} \nDatos test: {} \n".format(
        len(X_train_), len(X_dev), len(X_test)
    ))
    classes = np.unique(y_train)
    n_classes = len(classes)
    print("Classes: {}".format(" | ".join(classes)))
    """Fin Parser"""

    """Tokenizer"""
    if token:
        # tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

        tknzr = TweetTokenizer()
        X_train = [tknzr.tokenize(x) for x in X_train_]
        foo = sum(X_train, [])
        vocab = np.unique(foo)
        tam_vocab = len(vocab)
        del foo
        print("Tamaño del vocabulario: {}".format(tam_vocab))

    else:
        X_train = X_train_
        return X_train, y_train, ids_train, X_dev, y_dev, ids_dev, X_test, ids_test


    """Fin Tokenizer"""
    return X_train, y_train, ids_train, X_dev, y_dev, ids_dev, X_test, ids_test, vocab
