import os
import sys

sys.path.insert(0, '../../')

from datetime import datetime
from tqdm import tqdm
import analysis as dp
import configparser
import itertools
import matplotlib.pyplot as plt
import numpy as onp
import numpy.random as npr
import pandas as pd
import pickle
import seaborn as sns
import shutil

from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA

from jax import jit, grad, partial, random, tree_util, vmap, lax
from jax import numpy as np
from jax.experimental import optimizers, stax
from jax.nn.initializers import orthogonal, zeros, glorot_normal, normal

import mimic
import flows


key = random.PRNGKey(0)

def shuffle(rng, arr):
    arr = onp.asarray(arr).copy()
    onp.random.shuffle(arr)
    return np.array(arr)

# -----------------------------------------------------------------------

"""
X = mimic.get_patient_matrix(
    'ADMISSIONS.csv',
    'DIAGNOSES_ICD.csv',
    'binary'
)
"""

X, _ = datasets.make_moons(n_samples=10000, noise=.05)

scaler = preprocessing.StandardScaler()
X = np.array(scaler.fit_transform(X))

temp_key, key = random.split(key)
X = shuffle(temp_key, X)

# -----------------------------------------------------------------------

config_file = 'experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file)
config = config['DEFAULT']

b1 = config.getfloat('b1')
b2 = config.getfloat('b2')
delta = config.getfloat('delta')
iterations = config.getint('iterations')
dataset = config['dataset']
flow = config['flow']
lr = config.getfloat('lr')
l2_norm_clip = config.getfloat('l2_norm_clip')
microbatch_size = config.getint('microbatch_size')
minibatch_size = config.getint('minibatch_size')
num_hidden = config.getint('num_hidden')
noise_multiplier = config.getfloat('noise_multiplier')
private = config.getboolean('private')
weight_decay = config.getfloat('weight_decay')

# -----------------------------------------------------------------------

print('Achieves ({}, {})-DP'.format(
    dp.epsilon(
        X.shape[0],
        minibatch_size,
        noise_multiplier,
        iterations,
        delta,
    ),
    delta,
))

# -----------------------------------------------------------------------

def MaskedDense(out_dim, mask, W_init=glorot_normal(), b_init=normal()):
    init_fun, _ = stax.Dense(out_dim, W_init, b_init)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return np.dot(inputs, W * mask) + b

    return init_fun, apply_fun


def get_made_mask(in_features, out_features, in_flow_features, mask_type=None):
    if mask_type == "input":
        in_degrees = np.arange(in_features) % in_flow_features
    else:
        in_degrees = np.arange(in_features) % (in_flow_features - 1)

    if mask_type == "output":
        out_degrees = np.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = np.arange(out_features) % (in_flow_features - 1)

    return np.transpose(np.expand_dims(out_degrees, -1) >= np.expand_dims(in_degrees, 0)).astype(np.float32)


def get_affine_coupling_net(input_shape, hidden_dim=64, act=stax.Relu):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
        act,
        stax.Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
        act,
        stax.Dense(input_shape[-1], W_init=orthogonal(), b_init=zeros),
    )


def get_affine_coupling_mask(input_shape):
    mask = onp.zeros(input_shape)
    mask[::2] = 1.0
    return mask


input_shape = X.shape[1:]
num_samples = X.shape[0]
num_inputs = input_shape[-1]

input_mask = get_made_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
hidden_mask = get_made_mask(num_hidden, num_hidden, num_inputs)
output_mask = get_made_mask(num_hidden, num_inputs * 2, num_inputs, mask_type="output")

joiner = MaskedDense(num_hidden, input_mask)

trunk = stax.serial(
    stax.Relu,
    MaskedDense(num_hidden, hidden_mask),
    stax.Relu,
    MaskedDense(num_inputs * 2, output_mask),
)

bijection = flows.Serial(
    flows.MADE(joiner, trunk, num_hidden),
    flows.ActNorm(),
    flows.InvertibleLinear(),
    flows.MADE(joiner, trunk, num_hidden),
    flows.ActNorm(),
    flows.InvertibleLinear(),
    flows.MADE(joiner, trunk, num_hidden),
    flows.ActNorm(),
    flows.InvertibleLinear(),
)

prior = flows.Normal()

init_fun = flows.Flow(bijection, prior)

temp_key, key = random.split(key)
params, log_pdf, sample = init_fun(temp_key, input_shape)

#step_size = optimizers.inverse_time_decay(1e-3, 0, .9)
opt_init, opt_update, get_params = optimizers.adam(lr)
opt_state = opt_init(params)


def loss(params, inputs):
    return -log_pdf(params, inputs).mean()


def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier, minibatch_size):
    def _clipped_grad(params, single_example_batch):
        single_example_batch = np.expand_dims(single_example_batch, 0)
        grads = grad(loss)(params, single_example_batch)
        nonempty_grads, tree_def = tree_util.tree_flatten(grads)
        total_grad_norm = np.linalg.norm([np.linalg.norm(neg.ravel()) for neg in nonempty_grads])
        divisor = lax.stop_gradient(np.amax((total_grad_norm / l2_norm_clip, 1.)))
        normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
        return tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)

    px_clipped_grad_fn = vmap(partial(_clipped_grad, params))
    std_dev = l2_norm_clip * noise_multiplier
    noise_ = lambda n: n + std_dev * random.normal(rng, n.shape)
    normalize_ = lambda n: n / float(minibatch_size)
    sum_ = lambda n: np.sum(n, 0)
    aggregated_clipped_grads = tree_util.tree_map(sum_, px_clipped_grad_fn(batch))
    noised_aggregated_clipped_grads = tree_util.tree_map(noise_, aggregated_clipped_grads)
    normalized_noised_aggregated_clipped_grads = tree_util.tree_map(normalize_, noised_aggregated_clipped_grads)
    return normalized_noised_aggregated_clipped_grads


@jit
def private_update(rng, i, opt_state, batch):
    params = get_params(opt_state)
    grads = private_grad(params, batch, rng, l2_norm_clip, noise_multiplier, minibatch_size)
    return opt_update(i, grads, opt_state)


@jit
def update(rng, i, opt_state, batch):
    params = get_params(opt_state)
    grads = grad(loss)(params, batch)
    return opt_update(i, grads, opt_state)


try:
    os.mkdir('out')
except OSError as error:
    pass

datetime_str = datetime_str = datetime.now().strftime('%b-%d-%Y_%I:%M:%S_%p')
output_dir = 'out/' + datetime_str + '/'
os.mkdir(output_dir)
shutil.copyfile(config_file, output_dir + 'experiment.ini')

shutil.copyfile('train.py', output_dir + 'train.py')

plt.hist2d(X[:, 0], X[:, 1], bins=100, range=((-2, 2), (-2, 2)))
plt.savefig(output_dir + 'samples.png')
plt.clf()

pbar = tqdm(range(iterations))

for iteration in pbar:
    epoch = iteration // (X.shape[0] // minibatch_size)
    batch_index = iteration % (X.shape[0] // minibatch_size)
    batch_index_start = batch_index * minibatch_size

    if batch_index == 0:
        temp_key, key = random.split(key)
        X = shuffle(temp_key, X)

    batch = X[batch_index_start:batch_index_start+minibatch_size]

    temp_key, key = random.split(key)
    if private:
        opt_state = private_update(temp_key, iteration, opt_state, batch)
    else:
        opt_state = update(temp_key, iteration, opt_state, batch)


    if batch_index == 0:
        params = get_params(opt_state)
        l = loss(params, X)
        pbar.set_description('{:.4f}'.format(l))

        if np.isnan(l).any():
            print('NaN occurred! Exiting.')
            break

    if batch_index == 0 and epoch % 500 == 0:
        iteration_dir = output_dir + str(iteration) + '/'
        os.mkdir(iteration_dir)

        temp_key, key = random.split(key)
        X_syn = onp.asarray(sample(temp_key, params, num_samples))

        plt.hist2d(X_syn[:, 0], X_syn[:, 1], bins=100, range=((-2, 2), (-2, 2)))
        plt.savefig(iteration_dir + 'samples.png')
        plt.clf()

        pickle.dump(params, open(iteration_dir + 'params.pickle', 'wb'))
