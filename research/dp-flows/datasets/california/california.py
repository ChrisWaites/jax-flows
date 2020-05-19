import numpy as np
import jax.numpy as jnp
from sklearn import datasets
from .. import utils


@utils.constant_seed
def get_datasets():
    X = datasets.fetch_california_housing(data_home='datasets/california/', download_if_missing=True).data
    np.random.shuffle(X)
    X = jnp.array(X)

    val_cutoff = int(0.8 * X.shape[0])
    test_cutoff = int(0.9 * X.shape[0])

    return X[val_cutoff:], X[val_cutoff:test_cutoff], X[test_cutoff:]

