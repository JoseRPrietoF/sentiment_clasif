
import numpy as np
import traditional_models
from tfidf import make_tfidf
from model_tfidf_tf import Model
import logging
from data import get_data
import traditional_models
from tfidf import make_tfidf

def prepare(fname):
    """
    Logging and arguments
    :return:
    """

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    # --- keep this logger at DEBUG level, until aguments are processed
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    fh = logging.FileHandler(fname, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # --- restore ch logger to INFO
    ch.setLevel(logging.INFO)

    return logger

if __name__ == "__main__":
    fname =  "prueba"
    min_ngram, up = 1,7
    min_df = 3
    max_features = 10000
    X_train, y_train, ids_train, X_dev, y_dev, ids_dev, X_test, id_test = get_data()
    X_train, X_dev, X_test = make_tfidf(X_train, X_dev, X_test, min_ngram,up,max_features=max_features, min_df=min_df)
    logger = prepare(fname)
    Model(X_train,
                 X_dev,
                 y_train,
                 y_dev,
                ids_train,
          ids_dev,
                 logger=logger,
                NUM_EPOCH=50
                 )
