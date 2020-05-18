from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
import jax.numpy as np

scaler = preprocessing.MinMaxScaler((-1, 1))

def get_datasets(val_prop=0.1):
    X = np.array(arff.loadarff('datasets/tamilnadu/eb.arff').data)
    val_cutoff = int(X.shape[0] * (1 - val_prop))

    X_train = scaler.fit_transform(X[:val_cutoff])
    X_val = scaler.transform(X[val_cutoff:])
    X = scaler.transform(X)

    return X, X_train, X_val

def postprocess(X):
    return scaler.inverse_transform(X)

