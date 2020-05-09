import numpy as np
from sklearn import datasets
from .. import utils


@utils.constant_seed
def get_datasets(val_prop=.1):
    X = datasets.make_circles(
        n_samples=60000,
        noise=0.075,
        factor=.6,
    )[0].astype(np.float32)

    val_cutoff = int(X.shape[0] * (1 - val_prop))

    return X, X[:val_cutoff], X[val_cutoff:]


def postprocess(X):
    return X
