import numpy as np
from sklearn import datasets
from .. import utils


@utils.constant_seed
def get_datasets(val_prop=.1, test_prop=.1):
    X = datasets.fetch_california_housing(data_home='datasets/california/', download_if_missing=True).data

    np.random.shuffle(X)

    val_start = int(X.shape[0] * (1 - (test_prop + val_prop)))
    val_end = int(X.shape[0] * (1 - test_prop))

    return X, X[:val_start], X[val_start:val_end], X[val_end:]

