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

import flows


init_fun, encode = stax.serial(
    stax.Dense(2),
)

temp_key, key = random.split(key)
_, encoder_params = init_fun(temp_key, (-1, 1071))

init_fun, decode = stax.serial(
    stax.Dense(1071),
    stax.Sigmoid
)

temp_key, key = random.split(key)
_, decoder_params = init_fun(temp_key, (-1, 2))

opt_init, opt_update, get_params = optimizers.adam(1e-3)
temp_key, key = random.split(key)
opt_state = opt_init((encoder_params, decoder_params))


def loss(params, inputs):
    encoder_params, decoder_params = params
    preds = decode(decoder_params, encode(encoder_params, inputs))
    return -(inputs * np.log(preds) + (1 - inputs) * np.log(1 - preds)).mean()

@jit
def step(i, opt_state, inputs):
    params = get_params(opt_state)
    gradient = grad(loss)(params, inputs)
    return opt_update(i, gradient, opt_state)

batch_size = 100
num_epochs = 10

itercount = itertools.count()
for epoch in range(num_epochs):
    temp_key, key = random.split(key)
    X = shuffle(X)

    for batch_index in tqdm(range(0, len(X), batch_size)):
        batch = X[batch_index:batch_index+batch_size]
        opt_state = step(next(itercount), opt_state, batch)

    params = get_params(opt_state)
    print(loss(params, X))

encode_params, _ = get_params(opt_state)


X_proj = encode(encode_params, X)
plt.hist2d(X_proj[:, 0], X_proj[:, 1], bins=100)
plt.savefig('train.png')
