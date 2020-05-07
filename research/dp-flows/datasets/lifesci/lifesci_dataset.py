from .. import utils
import pandas as pd
import numpy as np

@utils.constant_seed
def get_datasets(val_prop=.1, test_prop=.1):
    df = pd.read_csv('datasets/lifesci/lifesci.csv')
    X = df.values.astype('float32')

    np.random.shuffle(X)

    val_start = int(X.shape[0] * (1 - (val_prop + test_prop)))
    val_end = int(X.shape[0] * (1 - test_prop))

    return X, X[:val_start], X[val_start:val_end], X[val_end:]


def postprocess(X):
    return X
