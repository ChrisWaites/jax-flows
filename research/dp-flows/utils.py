from datasets import *
from jax.experimental import optimizers
import jax.random as random
import matplotlib.pyplot as plt
import numpy as onp
import os
import pickle
import seaborn as sns


def make_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        pass


def plot_dist(X, path):
    plt.hist2d(X[:, 0], X[:, 1], bins=100, range=((-1.05, 1.05), (-1.05, 1.05)))
    plt.savefig(path)
    plt.clf()


def plot_marginals(X, path, overlay=None):
    for dim in range(X.shape[1]):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist(X[:, dim], color='red', alpha=0.5, bins=30, range=(-1.05, 1.05), label='Synthetic')
        if not (overlay is None):
            ax.hist(overlay[:, dim], color='blue', alpha=0.2, bins=30, range=(-1.05, 1.05), label='Real')
        plt.savefig(path + str(dim) + '.png')
        plt.clf()


def plot_loss(train_losses, val_losses, path):
    plt.plot(range(len(train_losses)), train_losses, c='red', label='Train')
    plt.plot(range(len(val_losses)), val_losses, c='blue', label='Validation')
    plt.savefig(path)
    plt.clf()


def plot_samples(key, params, sample, X, output_dir):
    temp_key, key = random.split(key)
    X_syn = onp.asarray(sample(temp_key, params, X.shape[0]))

    if X_syn.shape[1] == 2:
        plot_dist(X_syn, output_dir + 'synthetic.png')

    if X.shape[1] <= 16:
        plot_marginals(X_syn, output_dir, overlay=X)


def dump_obj(obj, output_path):
    pickle.dump(obj, open(output_path, 'wb'))


def log(key, params, sample, X, output_dir, train_losses=None, val_losses=None, epsilons=None):
    make_dir(output_dir)
    dump_obj(params, output_dir + 'params.pkl')
    if train_losses:
        dump_obj(train_losses, output_dir + 'train_losses.pkl')
    if val_losses:
        dump_obj(val_losses, output_dir + 'val_losses.pkl')
    if epsilons:
        dump_obj(epsilons, output_dir + 'epsilons.pkl')
    plot_samples(key, params, sample, X, output_dir)


def get_optimizer(optimizer, sched, b1=0.9, b2=0.999):
    if optimizer.lower() == 'adagrad':
        return optimizers.adagrad(sched)
    elif optimizer.lower() == 'adam':
        return optimizers.adam(sched, b1, b2)
    elif optimizer.lower() == 'rmsprop':
        return optimizers.rmsprop(sched)
    elif optimizer.lower() == 'momentum':
        return optimizers.momentum(sched, 0.9)
    elif optimizer.lower() == 'sgd':
        return optimizers.sgd(sched)
    else:
        raise Exception('Invalid optimizer: {}'.format(optimizer))


def get_datasets(dataset):
    return {
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
        'olivetti': olivetti.get_datasets,
        'pinwheel': pinwheel.get_datasets,
        'spam': spam.get_datasets,
    }[dataset]()
