import numpy as np
from .. import utils


@utils.constant_seed
def get_datasets(val_prop=.1, test_prop=.1):
    dataset = np.r_[
        np.random.randn(10000, 2) * 0.2 + np.array([3, 1]),
        np.random.randn(10000, 2) * 0.2 + np.array([3 - 1.414, 3 - 1.414]),
        np.random.randn(10000, 2) * 0.2 + np.array([1, 3]),
        np.random.randn(10000, 2) * 0.2 + np.array([3 - 1.414, 3 + 1.414]),
        np.random.randn(10000, 2) * 0.2 + np.array([3, 5]),
        np.random.randn(10000, 2) * 0.2 + np.array([3 + 1.414, 3 + 1.414]),
        np.random.randn(10000, 2) * 0.2 + np.array([5, 3]),
        np.random.randn(10000, 2) * 0.2 + np.array([3 + 1.414, 3 - 1.414]),
    ].astype(np.float32)

    np.random.shuffle(dataset)

    val_start = int(dataset.shape[0] * (1 - (test_prop + val_prop)))
    val_end = int(dataset.shape[0] * (1 - test_prop))

    train_dataset = dataset[:val_start]
    val_dataset = dataset[val_start:val_end]
    test_dataset = dataset[val_end:]

    return dataset, train_dataset, val_dataset, test_dataset


def postprocess(X):
    return X
