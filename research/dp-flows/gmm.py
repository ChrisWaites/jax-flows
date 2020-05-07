import os
import sys

sys.path.insert(0, '../../')

from datetime import datetime
from tqdm import tqdm
import configparser
import itertools
import matplotlib.pyplot as plt
import numpy as onp
import numpy.random as npr
import pandas as pd
import pickle
import seaborn as sns
import shutil

from sklearn import datasets, preprocessing, mixture
from sklearn.decomposition import PCA

from jax import jit, grad, partial, random, tree_util, vmap, lax
from jax import numpy as np
from jax.experimental import optimizers, stax
from jax.nn.initializers import orthogonal, zeros, glorot_normal, normal

import flows
from datasets import *
from dp import compute_eps_poisson, compute_eps_uniform
import plotting


def shuffle(rng, arr):
    onp.random.seed(rng[1])
    arr = onp.asarray(arr).copy()
    onp.random.shuffle(arr)
    return np.array(arr)


def main(config):
    print(dict(config))

    dataset = config['dataset']

    _, X, X_val, _ = {
        'adult': adult.get_datasets,
        'california': california.get_datasets,
        'checkerboard': checkerboard.get_datasets,
        'circles': circles.get_datasets,
        'credit': credit.get_datasets,
        'gaussian': gaussian.get_datasets,
        'gowalla': gowalla.get_datasets,
        'lifesci': lifesci.get_datasets,
        'mimic': mimic.get_datasets,
        'moons': moons.get_datasets,
        'pinwheel': pinwheel.get_datasets,
        'spam': spam.get_datasets,
    }[dataset]()

    #scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler((-1., 1.))
    X = np.array(scaler.fit_transform(X))
    X_val = np.array(scaler.transform(X_val))

    input_shape = X.shape[1:]
    num_samples = X.shape[0]
    num_inputs = input_shape[-1]

    GMM = mixture.GaussianMixture(n_components=7)
    GMM.fit(X)

    nll = -GMM.score_samples(X_val).mean()
    X_syn = GMM.sample(X.shape[0])[0]

    try:
        os.mkdir('out')
    except OSError as error:
        pass

    try:
        os.mkdir('out/' + dataset)
    except OSError as error:
        pass

    if X.shape[1] == 2:
        plt.hist2d(X[:, 0], X[:, 1], bins=100)#, range=((-2, 2), (-2, 2)))
        plt.savefig('out/' + dataset + '/' + 'data.png')
        plt.clf()

        plt.hist2d(X_syn[:, 0], X_syn[:, 1], bins=100)#, range=((-2, 2), (-2, 2)))
        plt.savefig('out/' + dataset + '/' + 'syn.png')
        plt.clf()

    plotting.plot_marginals(X_syn, 'out/' + dataset + '/', overlay=X)

    return {'nll': (nll, 0.)}


if __name__ == '__main__':
    config_file = 'experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    print('Best validation loss: {}'.format(main(config)))
