import jax.numpy as np
from sklearn import preprocessing
from .. import utils

scaler = preprocessing.MinMaxScaler((-1, 1))

@utils.constant_seed
def get_datasets():
    X = []
    for line in open('datasets/road/3D_spatial_network.txt', 'r'):
        X.append(list(map(float, line.split(','))))
    X = np.array(X)

    val_cutoff = int(0.8 * X.shape[0])
    test_cutoff = int(0.9 * X.shape[0])

    X_train = scaler.fit_transform(X[:val_cutoff])
    X_val = scaler.transform(X[val_cutoff:test_cutoff])
    X_test = scaler.transform(X[test_cutoff:])

    return X_train, X_val, X_test

def postprocess(X):
    return scaler.inverse_transform(X)

