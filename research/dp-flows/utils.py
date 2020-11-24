from datasets import *
from jax.experimental import optimizers
from dp import moments_accountant, gdp_accountant
import jax.random as random
import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
import os
import pickle
import flows
from itertools import islice


def make_dir(path):
  try:
    os.mkdir(path)
  except OSError as error:
    pass

def plot_dist(X, path, bounds=((-1.05, 1.05), (-1.05, 1.05))):
  plt.hist2d(X[:, 0], X[:, 1], bins=100, range=bounds)
  plt.savefig(path, dpi=1200)
  plt.close('all')

def plot_marginals(X, path, overlay=None):
  for dim in range(X.shape[1]):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(X[:, dim], color='orange', alpha=0.5, bins=100, range=(-1.05, 1.05), label='Synthetic')
    if not (overlay is None):
      ax.hist(overlay[:, dim], color='blue', alpha=0.2, bins=100, range=(-1.05, 1.05), label='Real')
    plt.legend()
    plt.grid(True)
    plt.savefig(path + str(dim) + '.png', dpi=1200)
    plt.close('all')

def plot_loss(train_losses, val_losses, path):
  plt.plot(range(len(train_losses)), train_losses, c='red', label='Train')
  plt.plot(range(len(val_losses)), val_losses, c='blue', label='Validation')
  plt.savefig(path, dpi=1200)
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

def log(params, output_dir, filename='params.pkl'):
  make_dir(output_dir)
  dump_obj(params, output_dir + filename)

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

def get_epsilon(private, composition, sampling, iteration, noise_multiplier, num_samples, minibatch_size, delta):
  if not private:
    return 1e8
  elif composition == 'gdp' and sampling == 'uniform':
    accountant = gdp_accountant.compute_eps_uniform
  elif composition == 'gdp' and sampling == 'poisson':
    accountant = gdp_accountant.compute_eps_poisson
  elif composition == 'ma' and sampling == 'uniform':
    accountant = moments_accountant.compute_eps_uniform
  elif composition == 'ma' and sampling == 'poisson':
    accountant = moments_accountant.compute_eps_poisson
  else:
    raise Exception('Invalid composition method: {}'.format(composition))
  return accountant(iteration, noise_multiplier, num_samples, minibatch_size, delta)

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
  elif sampling == 'uniform':
    # Uniform subsampling
    temp_key, key = random.split(key)
    X = random.permutation(temp_key, X)
    return X[:minibatch_size], X
  elif sampling == 'poisson':
    # Poisson subsampling
    temp_key, key = random.split(key)
    whether = random.uniform(temp_key, (X.shape[0],)) < (minibatch_size / X.shape[0])
    return X[whether], X
  else:
    raise Exception('Invalid sampling method: {}'.format(sampling))

dataset_map = {
  'lifesci': lifesci,
}

def get_datasets(dataset):
  return dataset_map[dataset].get_datasets()

def take(n, iterable):
  "Return first n items of the iterable as a list"
  return list(islice(iterable, n))

def get_gmm_params(path):
  mat = loadmat(path)
  means = mat['model']['cpd'][0][0][0][0][0].transpose()
  covariances = mat['model']['cpd'][0][0][0][0][1].transpose()
  weights = mat['model']['mixWeight'][0][0][0]
  return means, covariances, weights

def get_prior(prior_type):
  if prior_type == 'normal':
    prior = flows.Normal()
  else:
    means, covariances, weights = get_gmm_params(prior)
    prior = flows.GMM(means, covariances, weights)
  return prior
