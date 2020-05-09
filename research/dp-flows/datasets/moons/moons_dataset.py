from .. import utils
from sklearn import datasets
import numpy as np


@utils.constant_seed
def get_datasets(val_prop=.1):
    X = datasets.make_moons(
        n_samples=60000,
        noise=0.05
    )[0].astype(np.float32)

    val_cutoff = int(X.shape[0] * (1 - val_prop))

    return X, X[:val_cutoff], X[val_cutoff:]


def postprocess(X):
    return X
