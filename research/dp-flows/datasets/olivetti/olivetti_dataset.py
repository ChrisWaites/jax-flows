import numpy as np
from sklearn import datasets
from .. import utils


@utils.constant_seed
def get_datasets(val_prop=.1, test_prop=.1):
    dataset = datasets.fetch_olivetti_faces('datasets/olivetti/').data.astype(np.float32)

    val_start = int(dataset.shape[0] * (1 - (test_prop + val_prop)))
    val_end = int(dataset.shape[0] * (1 - test_prop))

    train_dataset = dataset[:val_start]
    val_dataset = dataset[val_start:val_end]
    test_dataset = dataset[val_end:]

    return dataset, train_dataset, val_dataset, test_dataset


def postprocess(X):
    return X
