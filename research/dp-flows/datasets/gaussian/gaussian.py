import numpy as np
import jax.numpy as jnp
from .. import utils


@utils.constant_seed
def get_datasets():
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
    X = jnp.array(X)

    val_cutoff = int(0.8 * X.shape[0])
    test_cutoff = int(0.9 * X.shape[0])

    return X[:val_cutoff], X[val_cutoff:test_cutoff], X[test_cutoff:]


def postprocess(X):
    return X
