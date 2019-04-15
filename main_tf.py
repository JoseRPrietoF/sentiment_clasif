
import numpy as np
import traditional_models
from tfidf import make_tfidf
import model_WE_tf
import model_tfidf_tf
import logging
from data import get_data
import traditional_models
from tfidf import make_tfidf
from utils import *

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

    MODEL = "FF"
    # MODEL = "CNN"
    # MODEL = "RNN"
    work_path = "work"
    create_structure(work_path)
    fname = "{}/prueba_{}".format(work_path, MODEL)

    if MODEL == "FF":
        min_ngram, up = 1, 9
        min_df = 3
        max_features = 50000
        X_train, y_train, ids_train, X_dev, y_dev, ids_dev, X_test, id_test = get_data()
        X_train, X_dev, X_test = make_tfidf(X_train, X_dev, X_test, min_ngram, up, max_features=max_features,
                                            min_df=min_df)
        logger = prepare(fname)
        layers = [256, 128, 64, 32]
        model_tfidf_tf.Model(texts_rep_train=X_train,
                             text_test_rep=X_test,
                             y_train=y_train,
                             y_dev=y_dev,
                             fnames_train=ids_train,
                             fnames_test=id_test,
                             text_dev_rep=X_dev,
                             fnames_dev=ids_dev,

                             MODEL=MODEL,
                             logger=logger,
                             NUM_EPOCH=300,
                             # layers=[1024, 512, 256, 128, 64],
                             layers=layers,
                             OPTIMIZER='adam',
                             path=work_path,
                             )
    else:
        X_train, y_train, ids_train, X_dev, y_dev, ids_dev, X_test, id_test, vocab = get_data(token=True)
        logger = prepare(fname)
        filters = [512,1024,1024]

        # EMBED_SIZE = 300
        EMBED_SIZE = 200
        # dir_word_embeddings = '/data2/jose/word_embedding/glove-sbwc.i25.vec'
        # dir_word_embeddings = '/data2/jose/word_embedding/fastext/fasttext-sbwc.vec'
        dir_word_embeddings = '/data2/jose/word_embedding/es_embeddings_200M_200d/embedding_file'

        model_WE_tf.Model(X_train,
                          X_dev,
                          y_train,
                          y_dev,
                          ids_train,
                          ids_dev,

                          NUM_LAYERS=3,
                          HIDDEN_UNITS=64,
                          logger=logger,
                          MODEL=MODEL,
                          NUM_EPOCH=30,
                          filters=filters,
                          OPTIMIZER='adam',
                          vocab=vocab,
                          EMBEDDING_DIM=EMBED_SIZE,
                          dir_word_embeddings=dir_word_embeddings,

                          )
