import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def make_tfidf(X_train, X_dev, X_test, min_ngram,up,max_features=None):
    if max_features:

        rep = TfidfVectorizer(ngram_range=(min_ngram, up), max_features=max_features)
    else:
        rep = TfidfVectorizer(ngram_range=(min_ngram, up))
    print(X_train[:5])
    texts_rep_train = rep.fit_transform(X_train)
    texts_rep_train = texts_rep_train.toarray()

    text_test_rep = rep.transform(X_test)
    text_dev_rep = rep.transform(X_dev)
    text_test_rep = text_test_rep.toarray()
    text_dev_rep = text_dev_rep.toarray()

    return texts_rep_train, text_dev_rep, text_test_rep

