import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer


def dummy_fun(doc):
    return TweetTokenizer().tokenize(doc)

def get_words_PN(path="data/ElhPolar_esV1.lex"):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    p,n = [],[]
    for line in lines:
        if not line.startswith("#") and len(line) > 5:
            word, sense = line.split()
            if sense == "positive":
                p.append(word.lower())
            else:
                n.append(word.lower())

    return p,n

def make_tfidf(X_train, X_dev, X_test, min_ngram,up,max_features=None, min_df=1, SOA=True):
    if SOA:
        print("Doing SOA")
        positive, negative = get_words_PN()
        adding = []
        for i in X_train:
            i = i.lower()
            p_count = 0
            for p in positive:
                p_count += i.count(p)
            n_count = 0
            for n in negative:
                n_count += i.count(n)

            adding.append([p_count, n_count])
        adding = np.array(adding)

        adding_dev = []
        for i in X_dev:
            i = i.lower()
            p_count = 0
            for p in positive:
                p_count += i.count(p)
            n_count = 0
            for n in negative:
                n_count += i.count(n)

            adding_dev.append([p_count, n_count])

        adding_test = []
        for i in X_test:
            i = i.lower()
            p_count = 0
            for p in positive:
                p_count += i.count(p)
            n_count = 0
            for n in negative:
                n_count += i.count(n)

            adding_test.append([p_count, n_count])

        adding_test = np.array(adding_test)
        print("SOA finished")

    if max_features is None:
        print("No limits")
        rep = TfidfVectorizer(ngram_range=(min_ngram, up), max_features=max_features, min_df=min_df, tokenizer=dummy_fun)
    else:
        rep = TfidfVectorizer(ngram_range=(min_ngram, up), min_df=min_df, tokenizer=dummy_fun)

    texts_rep_train = rep.fit_transform(X_train)
    texts_rep_train = texts_rep_train.toarray()

    text_test_rep = rep.transform(X_test)
    text_dev_rep = rep.transform(X_dev)
    text_test_rep = text_test_rep.toarray()
    text_dev_rep = text_dev_rep.toarray()

    texts_rep_train = np.concatenate([texts_rep_train, adding], axis=1)
    text_dev_rep = np.concatenate([text_dev_rep, adding_dev], axis=1)
    text_test_rep = np.concatenate([text_test_rep, adding_test], axis=1)

    return texts_rep_train, text_dev_rep, text_test_rep

