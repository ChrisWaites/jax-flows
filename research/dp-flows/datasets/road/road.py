import jax.numpy as np
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler((-1, 1))

def get_datasets(val_prop=0.1):
    X = []
    for line in open('datasets/road/3D_spatial_network.txt', 'r'):
        X.append(list(map(float, line.split(','))))
    X = np.array(X)

    val_cutoff = int(X.shape[0] * (1 - val_prop))

    X_train = scaler.fit_transform(X[:val_cutoff])
    X_val = scaler.transform(X[val_cutoff:])
    X = scaler.transform(X)

    return X, X_train, X_val

def postprocess(X):
    return scaler.inverse_transform(X)

