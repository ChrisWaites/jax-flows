import numpy as np
from .. import utils

@utils.constant_seed
def get_datasets(val_prop=0.1, test_prop=0.1):
    X = []
    with open('datasets/credit/german.data-numeric', 'r') as f:
        for line in f:
            line = line.split(' ')
            X.append(line)
    X = np.array(X)

    val_start = int(X.shape[0] * (1 - (val_prop + test_prop)))
    val_end = int(X.shape[0] * (1 - test_prop))

    return X, X[:val_start], X[val_start:val_end], X[val_end:]
