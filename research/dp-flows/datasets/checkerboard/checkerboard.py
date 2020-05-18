import numpy as np
import jax.numpy as jnp
from .. import utils

def block(x_low, x_high, y_low, y_high):
    return np.column_stack((
        np.random.uniform(x_low, x_high, 7500),
        np.random.uniform(y_low, y_high, 7500),
    ))

@utils.constant_seed
def get_datasets(val_prop=0.1, test_prop=0.1):
    dataset = np.r_[
        block(-2, -1, 1, 2),
        block(-2, -1, -1, 0),
        block(-1, 0, 0, 1),
        block(-1, 0, -2, -1),
        block(0, 1, 1, 2),
        block(0, 1, -1, 0),
        block(1, 2, 0, 1),
        block(1, 2, -2, -1),
    ].astype(np.float32)

    np.random.shuffle(dataset)
    dataset = jnp.array(dataset)

    val_start = int(dataset.shape[0] * (1 - (test_prop + val_prop)))
    val_end = int(dataset.shape[0] * (1 - test_prop))

    train_dataset = dataset[:val_start]
    val_dataset = dataset[val_start:val_end]
    test_dataset = dataset[val_end:]

    return dataset, train_dataset, val_dataset, test_dataset
