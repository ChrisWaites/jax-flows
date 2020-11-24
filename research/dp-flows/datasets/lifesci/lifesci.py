from .. import utils
import pandas as pd
import jax.numpy as np

@utils.constant_seed
def get_datasets(split=None):
    return np.array(pd.read_csv('datasets/lifesci/lifesci.csv').values.astype('float32'))

def postprocess(X):
    return X
