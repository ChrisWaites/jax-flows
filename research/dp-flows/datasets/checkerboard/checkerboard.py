import numpy as np
import jax.numpy as jnp
from .. import utils

def block(x_low, x_high, y_low, y_high):
    return np.column_stack((
        np.random.uniform(x_low, x_high, 7500),
        np.random.uniform(y_low, y_high, 7500),
    ))

@utils.constant_seed
def get_datasets():
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

    val_start = int(0.8 * dataset.shape[0])
    val_end = int(0.9 * dataset.shape[0])

    train_dataset = dataset[:val_start]
    val_dataset = dataset[val_start:val_end]
    test_dataset = dataset[val_end:]

    return dataset, train_dataset, val_dataset, test_dataset
