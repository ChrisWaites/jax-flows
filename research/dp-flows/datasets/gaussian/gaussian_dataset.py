import numpy as np
from .. import utils


@utils.constant_seed
def get_datasets(val_prop=.1):
    X = np.r_[
        np.random.randn(10000, 2) * 0.2 + np.array([3, 1]),
        np.random.randn(10000, 2) * 0.2 + np.array([3 - 1.414, 3 - 1.414]),
        np.random.randn(10000, 2) * 0.2 + np.array([1, 3]),
        np.random.randn(10000, 2) * 0.2 + np.array([3 - 1.414, 3 + 1.414]),
        np.random.randn(10000, 2) * 0.2 + np.array([3, 5]),
        np.random.randn(10000, 2) * 0.2 + np.array([3 + 1.414, 3 + 1.414]),
        np.random.randn(10000, 2) * 0.2 + np.array([5, 3]),
        np.random.randn(10000, 2) * 0.2 + np.array([3 + 1.414, 3 - 1.414]),
    ].astype(np.float32)

    np.random.shuffle(X)

    val_cutoff = int(X.shape[0] * (1 - val_prop))

    return X, X[:val_cutoff], X[val_cutoff:]


def postprocess(X):
    return X
