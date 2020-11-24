import os
import sys

sys.path.insert(0, '../../')

from datetime import datetime
from jax import jit, grad, partial, random, tree_util, vmap, lax, ops
from jax import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten
from sklearn import model_selection, mixture, preprocessing, preprocessing
from tqdm import tqdm
import numpy as onp
import pickle
import shutil
import itertools

import dp
import flow_utils
import flows
import utils

from scipy.io import loadmat
import glob
from config import *


def log_params_cond(log_params, iteration):
  return log_params \
    and (iteration == 1 \
    or (iteration < 5000 and iteration % 500 == 0) \
    or iteration % log_params == 0)

def clipped_grad(params, l2_norm_clip, single_example_batch):
  """Evaluate gradient for a single-example batch and clip its grad norm."""
  grads = grad(loss)(params, single_example_batch)
  nonempty_grads, tree_def = tree_flatten(grads)
  total_grad_norm = np.linalg.norm([np.linalg.norm(neg.ravel()) for neg in nonempty_grads])
  divisor = np.max(np.array([total_grad_norm / l2_norm_clip, 1.]))
  normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
  return tree_unflatten(tree_def, normalized_nonempty_grads)

def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier, batch_size):
  """Return differentially private gradients for params, evaluated on batch."""
  clipped_grads = vmap(clipped_grad, (None, None, 0))(params, l2_norm_clip, batch)
  clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
  aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
  rngs = random.split(rng, len(aggregated_clipped_grads))
  noised_aggregated_clipped_grads = [g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape) for r, g in zip(rngs, aggregated_clipped_grads)]
  normalized_noised_aggregated_clipped_grads = [g / batch_size for g in noised_aggregated_clipped_grads]
  return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)

def loss(params, inputs):
  """Negative log-likelihood."""
  return -log_pdf(params, inputs).mean()

@jit
def private_update(rng, i, opt_state, batch):
  batch = np.expand_dims(batch, 1)
  params = get_params(opt_state)
  grads = private_grad(params, batch, rng, l2_norm_clip, noise_multiplier, minibatch_size)
  return opt_update(i, grads, opt_state)

@jit
def update(rng, i, opt_state, batch):
  params = get_params(opt_state)
  grads = grad(loss)(params, batch)
  return opt_update(i, grads, opt_state)

if __name__ == '__main__':
  key = random.PRNGKey(0)

  # Create dataset
  X_full = utils.get_datasets(dataset)

  kfold = model_selection.KFold(pieces, shuffle=True, random_state=0)
  for fold_iter, (idx_train, idx_test) in enumerate(utils.take(pieces_to_run, kfold.split(X_full))):
    X, X_test = X_full[idx_train], X_full[idx_test]

    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    delta = 1. / (X.shape[0] ** 1.1)

    print('X: {}'.format(X.shape))
    print('X test: {}'.format(X_test.shape))
    print('Delta: {}'.format(delta))

    # Create flow
    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))

    # Remove previous directory to avoid ambiguity
    if log_params and overwrite:
      try:
        shutil.rmtree(output_dir)
      except: pass

    # Create experiment directory
    output_dir_tokens = ['out', dataset, 'flows', experiment, str(fold_iter)]
    output_dir = ''
    for ext in output_dir_tokens:
      output_dir += ext + '/'
      utils.make_dir(output_dir)

    if save_dataset:
      onp.savetxt(open(output_dir + 'train.csv', 'w'), np.asarray(X))
      onp.savetxt(open(output_dir + 'test.csv', 'w'), np.asarray(X_test))

    # Create prior
    prior = utils.get_prior(prior_type)

    # Create flow
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    params, log_pdf, sample = init_fun(temp_key, X.shape[1])

    pbar_range = range(1, iterations + 1)

    # Create directories and load previous params if applicable
    if log_params:
      # Loads last params if any exist
      experiment_dir = '/'.join(output_dir_tokens[:-1]) + '/'
      param_dirs = sorted([int(path) for path in os.listdir(experiment_dir) if os.path.isdir(output_dir + path)])
      if len(param_dirs) > 0:
        last_iteration = param_dirs[-1]
        print('Loading params from {}...'.format(str(last_iteration) + '/params.pkl'))
        params = pickle.load(open(output_dir + str(last_iteration) + '/params.pkl', 'rb'))
        pbar_range = range(last_iteration, last_iteration + iterations)

      # Log all train files
      shutil.copyfile('experiment.ini', experiment_dir + 'experiment.ini')
      shutil.copyfile('train.py', experiment_dir + 'train.py')
      shutil.copyfile('flow_utils.py', experiment_dir + 'flow_utils.py')

      # Log serialized prior
      if prior_type != 'normal':
        shutil.copyfile('centers.mat', experiment_dir + 'centers.mat')
        utils.log(means, experiment_dir + 'means.pkl')
        utils.log(covariances, experiment_dir + 'covariances.pkl')
        utils.log(weights, experiment_dir + 'weights.pkl')

    # Create optimizer
    scheduler = utils.get_scheduler(lr, lr_schedule)
    opt_init, opt_update, get_params = utils.get_optimizer(optimizer, scheduler, b1, b2)
    opt_state = opt_init(params)
    update_fn = private_update if private else update

    best_test_params, best_test_loss = None, None
    pbar = tqdm(pbar_range)
    for iteration in pbar:
      batch, X = utils.get_batch(sampling, key, X, minibatch_size, iteration)

      # Possible with Poisson subsampling
      if batch.shape[0] == 0:
        continue

      # Perform model update
      temp_key, key = random.split(key)
      opt_state = update_fn(temp_key, iteration, opt_state, batch)

      # Log params
      if log_params_cond(log_params, iteration):
        params = get_params(opt_state)
        utils.log(params, output_dir + str(iteration) + '/')

      # Update progress bar
      if iteration % log_performance == 0:
        # Calculate privacy loss
        epsilon = utils.get_epsilon(
          private, composition, sampling, iteration,
          noise_multiplier, X.shape[0], minibatch_size, delta
        )

        # Calculate losses
        params = get_params(opt_state)
        train_loss, test_loss = loss(params, X), loss(params, X_test)

        # Exit if NaN, as all has failed...
        if np.isnan(train_loss).any():
          break

        # Update best test model thus far
        if best_test_loss is None or test_loss < best_test_loss:
          best_test_loss = test_loss
          best_test_params = params
          if log_params:
            utils.log(best_test_params, output_dir, 'test_params.pkl')

        # Update progress bar
        pbar_text = 'Train: {:.3f} Test: {:.3f} Best Test: {:.3f} ε: {:.3f}'.format(
          -train_loss, -test_loss, -best_test_loss, epsilon,
        )

        pbar.set_description(pbar_text)
