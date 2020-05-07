from datasets import *
import matplotlib.pyplot as plt
import os
import seaborn as sns


def plot_dist(X, path):
    plt.hist2d(X[:, 0], X[:, 1], bins=100, range=((-1.05, 1.05), (-1.05, 1.05)))
    plt.savefig(path)
    plt.clf()


def plot_marginals(X, path, overlay=None):
    for dim in range(X.shape[1]):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist(X[:, dim], color='red', alpha=0.5, bins=30, range=(-1.05, 1.05))
        if not (overlay is None):
            ax.hist(overlay[:, dim], color='blue', alpha=0.2, bins=30, range=(-1.05, 1.05))
        plt.savefig(path + str(dim) + '.png')
        plt.clf()


def plot_loss(train_losses, val_losses, path):
    plt.plot(range(len(train_losses)), train_losses, c='red')
    plt.plot(range(len(val_losses)), val_losses, c='blue')
    plt.savefig(path)
    plt.clf()


def make_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        pass


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
        'pinwheel': pinwheel.get_datasets,
        'spam': spam.get_datasets,
    }[dataset]()

