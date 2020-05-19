from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
import jax.numpy as np

from .. import utils

scaler = preprocessing.MinMaxScaler((-1, 1))

@utils.constant_seed
def get_datasets():
    X = np.array(arff.loadarff('datasets/tamilnadu/eb.arff').data)
    val_cutoff = int(0.8 * X.shape[0])
    test_cutoff = int(0.9 * X.shape[0])

    X_train = scaler.fit_transform(X[:val_cutoff])
    X_val = scaler.transform(X[val_cutoff:test_cutoff])
    X_test = scaler.transform(X[test_cutoff:])

    return X_train, X_val, X_test

def postprocess(X):
    return scaler.inverse_transform(X)
