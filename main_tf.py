import tensorflow as tf
from data import process
from models import CNN
from models import FF
from models import RNN
from data.prepare_text import prepare_data
from utils import train_ops
from utils import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from utils import sesions
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

from data import get_data
import numpy as np
import traditional_models
from tfidf import make_tfidf












if __name__ == "__main__":
    min_ngram, up = 1,7
    min_df = 3
    max_features = None
    X_train, y_train, X_dev, y_dev, X_test = get_data()
    X_train, X_dev, X_test = make_tfidf(X_train, X_dev, X_test, min_ngram,up,max_features=max_features, min_df=min_df)
