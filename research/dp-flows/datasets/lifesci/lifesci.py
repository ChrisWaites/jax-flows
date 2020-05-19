from .. import utils
import pandas as pd
import jax.numpy as np
from sklearn import decomposition

pca = decomposition.PCA(n_components=2)

@utils.constant_seed
def get_datasets(split=None):
    if split:
        X_train = pd.read_csv('datasets/lifesci/train/{}.csv'.format(split)).values.astype('float32')
        X_test = pd.read_csv('datasets/lifesci/test/{}.csv'.format(split)).values.astype('float32')
    else:
        X_train = pd.read_csv('datasets/lifesci/lifesci_train.csv').values.astype('float32')
        X_test = pd.read_csv('datasets/lifesci/lifesci_test.csv').values.astype('float32')
    pca.fit(X_train)
    return np.array(X_train), np.array(X_test), np.array(X_test)

def postprocess(X):
    return X
