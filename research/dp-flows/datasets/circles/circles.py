import jax.numpy as np
from sklearn import datasets
from .. import utils


@utils.constant_seed
def get_datasets():
    X = np.array(datasets.make_circles(
        n_samples=60000,
        noise=0.075,
        factor=.6,
    )[0])

    val_cutoff = int(0.8 * X.shape[0])
    test_cutoff = int(0.9 * X.shape[0])

    return X[:val_cutoff], X[val_cutoff:test_cutoff], X[test_cutoff:]


def postprocess(X):
    return X
