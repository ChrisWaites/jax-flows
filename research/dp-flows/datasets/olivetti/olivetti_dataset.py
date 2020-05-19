import jax.numpy as np
from sklearn import datasets
from .. import utils


@utils.constant_seed
def get_datasets():
    dataset = np.array(datasets.fetch_olivetti_faces('datasets/olivetti/').data)

    val_cutoff = int(0.8 * dataset.shape[0])
    test_cutoff = int(0.9 * dataset.shape[0])

    X_train = dataset[:val_cutoff]
    X_val = dataset[val_cutoff:test_cutoff]
    X_test = dataset[test_cutoff:]

    return X_train, X_val, X_test


def postprocess(X):
    return X
