import os
import sys

sys.path.insert(0, '../')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp
import pandas as pd
import configparser
from datetime import datetime
import shutil
from tqdm import tqdm
import itertools
import pickle

from jax import numpy as np
import jax.random as random
from jax import jit, grad
from jax.experimental import optimizers, stax
from jax.nn.initializers import orthogonal, zeros

import mimic
import analysis as dp

from sklearn.decomposition import PCA
from sklearn import datasets, preprocessing

import flows


key = random.PRNGKey(0)


def shuffle(arr):
    #temp_key, key = random.split(key)
    #X = random.shuffle(temp_key, X)
    arr = onp.asarray(arr).copy()
    onp.random.shuffle(arr)
    return np.array(arr)

X = mimic.get_patient_matrix(
    'ADMISSIONS.csv',
    'DIAGNOSES_ICD.csv',
    'binary'
)

"""
scaler = preprocessing.StandardScaler()
X, _ = datasets.make_moons(n_samples=10000, noise=.05)
X = scaler.fit_transform(X)

X = np.array(X)
X = shuffle(X)

plt.hist2d(X[:, 0], X[:, 1], bins=100)
plt.savefig('syn.png')
"""


# -----------------------------------------------------------------------


config_file = 'experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file)
config = config['DEFAULT']

b1 = float(config['b1'])
b2 = float(config['b2'])
delta = float(config['delta'])
iterations = int(config['iterations'])
dataset = config['dataset']
flow = config['flow']
lr = float(config['lr'])
l2_norm_clip = float(config['l2_norm_clip'])
microbatch_size = int(config['microbatch_size'])
minibatch_size = int(config['minibatch_size'])
num_blocks = int(config['num_blocks'])
num_hidden = int(config['num_hidden'])
noise_multiplier = float(config['noise_multiplier'])
weight_decay = float(config['weight_decay'])

input_shape = X.shape[1:]


def net(input_shape=(3,), hidden_dim=64, act=stax.Relu):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
        act,
        stax.Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
        act,
        stax.Dense(input_shape[-1], W_init=orthogonal(), b_init=zeros),
    )


def mask(input_shape=(3,)):
    mask = onp.zeros(input_shape)
    mask[::2] = 1.0
    return mask


bijection = flows.serial(
    flows.Shuffle(),
    flows.MADE(),
    flows.Shuffle(),
    flows.MADE(),
    flows.Shuffle(),
    flows.MADE(),
    flows.Shuffle(),
    flows.MADE(),
    flows.Shuffle(),
)

prior = flows.Normal()

init_fun = flows.Flow(bijection, prior)

temp_key, key = random.split(key)
params, log_pdf, sample = init_fun(temp_key, input_shape)


opt_init, opt_update, get_params = optimizers.adam(1e-4)
temp_key, key = random.split(key)
opt_state = opt_init(params)


def loss(params, inputs):
    return -log_pdf(params, inputs).mean()


@jit
def step(i, opt_state, inputs):
    params = get_params(opt_state)
    gradient = grad(loss)(params, inputs)
    return opt_update(i, gradient, opt_state)


batch_size = 500
num_epochs = 1000

pbar = tqdm(range(num_epochs))

itercount = itertools.count()
for epoch in pbar:
    temp_key, key = random.split(key)
    X = shuffle(X)

    for batch_index in range(0, len(X), batch_size):
        batch = X[batch_index:batch_index+batch_size]
        opt_state = step(next(itercount), opt_state, batch)

    params = get_params(opt_state)
    temp_key, key = random.split(key)
    loss_i = loss(params, X)
    pbar.set_description('{:.4f}'.format(loss_i))

    if np.isnan(loss_i).any():
        print('NaN occurred! Exiting.')
        break

pickle.dump(params, open('params.pickle', 'wb'))

# params = pickle.load(open('params.pickle', 'rb'))

params = get_params(opt_state)
temp_key, key = random.split(key)
num_samples = X.shape[0]
X_syn = sample(temp_key, params, 1000)
#X_syn_proj = onp.asarray(encode(encoder_params, X_syn))
X_syn_proj = onp.asarray(X_syn)


plt.hist2d(X_syn_proj[:, 0], X_syn_proj[:, 1], bins=100)
plt.savefig('syn.png')


"""
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

try:
    os.mkdir('out')
except OSError as error:
    pass

datetime_str = datetime_str = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
output_dir = 'out/' + datetime_str + '/'
os.mkdir(output_dir)
shutil.copyfile(config_file, output_dir + 'experiment.ini')
"""
