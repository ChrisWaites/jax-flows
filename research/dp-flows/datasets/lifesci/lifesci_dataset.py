from .. import utils
import pandas as pd
import numpy as np

@utils.constant_seed
def get_datasets(val_prop=.1):
    df_train = pd.read_csv('datasets/lifesci/lifesci_train.csv')
    X_train = df_train.values.astype('float32')

    df_test = pd.read_csv('datasets/lifesci/lifesci_test.csv')
    X_test = df_test.values.astype('float32')

    return X_train, X_train, X_test


def postprocess(X):
    return X
