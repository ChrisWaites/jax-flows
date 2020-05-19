from datasets import *
from jax.experimental import optimizers
import dp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
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


def get_scheduler(lr, lr_schedule):
    if lr_schedule == 'constant':
        return lr
    elif lr_schedule == 'exponential':
        return lambda i: lr * (0.99995 ** i)
    else:
        raise Exception('Invalid lr scheduler: {}'.format(lr_scheduler))
    

def get_epsilon(sampling, composition, private, iteration, noise_multiplier, n, minibatch_size, delta):
    if not private:
        return 999999.
    elif composition == 'gdp':
        if sampling == 'poisson':
            return dp.compute_eps_poisson(iteration, noise_multiplier, n, minibatch_size, delta)
        elif sampling == 'uniform':
            return dp.compute_eps_uniform(iteration, noise_multiplier, n, minibatch_size, delta)
        else:
            raise Exception('Invalid sampling method {} for composition {}.'.format(sampling, composition))
    elif composition == 'ma':
        if sampling == 'poisson':
            return dp.epsilon(n, minibatch_size, noise_multiplier, iteration, delta)
        else:
            raise Exception('Invalid sampling method {} for composition {}.'.format(sampling, composition))
    else:
        raise Exception('Invalid composition method: {}'.format(composition))


def get_batch(sampling, key, X, minibatch_size, iteration):
    if sampling == 'batch':
        # Calculate epoch from iteration
        epoch = iteration // (X.shape[0] // minibatch_size)
        batch_index = iteration % (X.shape[0] // minibatch_size)
        batch_index_start = batch_index * minibatch_size
        # Regular batching
        if batch_index == 0:
            temp_key, key = random.split(key)
            X = random.permutation(temp_key, X)
        return X[batch_index_start:batch_index_start+minibatch_size], X
    elif sampling == 'poisson':
        # Poisson subsampling
        temp_key, key = random.split(key)
        whether = random.uniform(temp_key, (X.shape[0],)) < (minibatch_size / X.shape[0])
        return X[whether], X
    elif sampling == 'uniform':
        # Uniform subsampling
        temp_key, key = random.split(key)
        X = random.permutation(temp_key, X)
        return X[:minibatch_size], X
    else:
        raise Exception('Invalid sampling method: {}'.format(sampling))


def get_datasets(dataset):
    return {
        'adult': adult,
        'bsds300': bsds300,
        'california': california,
        'checkerboard': checkerboard,
        'circles': circles,
        'credit': credit,
        'gas': gas,
        'gaussian': gaussian,
        'gowalla': gowalla,
        'hepmass': hepmass,
        'lifesci': lifesci,
        'mimic': mimic,
        'miniboone': miniboone,
        'moons': moons.get_datasets,
        'olivetti': olivetti,
        'pinwheel': pinwheel,
        'power': power,
        'road': road,
        'spam': spam,
        'tamilnadu': tamilnadu,
    }[dataset].get_datasets()


def get_pca(dataset):
    return {
        'lifesci': lifesci,
    }[dataset].pca


