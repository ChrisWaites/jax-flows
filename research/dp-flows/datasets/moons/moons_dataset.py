from .. import utils
from sklearn import datasets
import numpy as np


@utils.constant_seed
def get_datasets(val_prop=.1, test_prop=.1):
    dataset = datasets.make_moons(
        n_samples=60000,
        noise=0.05
    )[0].astype(np.float32)

    val_start = int(dataset.shape[0] * (1 - (test_prop + val_prop)))
    val_end = int(dataset.shape[0] * (1 - test_prop))

    train_dataset = dataset[:val_start]
    val_dataset = dataset[val_start:val_end]
    test_dataset = dataset[val_end:]

    return dataset, train_dataset, val_dataset, test_dataset


def postprocess(X):
    return X
