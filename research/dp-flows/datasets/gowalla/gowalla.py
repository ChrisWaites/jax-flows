import numpy as np
import jax.numpy as jnp
from .. import utils

@utils.constant_seed
def get_datasets(val_prop=0.1):
    X = []
    with open('datasets/gowalla/gowalla.txt', 'r') as f:
        for line in f:
            line = line.split('\t')
            lat, lon = float(line[2]), float(line[3])
            X.append((lat, lon))
    X = np.array(X)

    np.random.shuffle(X)
    X = jnp.array(X[:60000])
    val_cutoff = int(X.shape[0] * (1 - val_prop))
    return X, X[:val_cutoff], X[val_cutoff:]
