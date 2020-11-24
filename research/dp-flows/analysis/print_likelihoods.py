import sys

sys.path.insert(0, './')
sys.path.insert(0, '../../')

from jax import random
import configparser
import jax.numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

import flows
import utils
import shutil


if __name__ == '__main__':
    flow_path  = 'out/lifesci/flows/private/' if len(sys.argv) == 1 else sys.argv[1]

    key = random.PRNGKey(0)

    config_file = flow_path + 'experiment.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']

    composition = config['composition'].lower()
    dataset = config['dataset'].lower()
    flow = config['flow'].lower()
    minibatch_size = int(config['minibatch_size'])
    noise_multiplier = float(config['noise_multiplier'])
    normalization = str(config['normalization']).lower() == 'true'
    num_blocks = int(config['num_blocks'])
    num_hidden = int(config['num_hidden'])
    private = str(config['private']).lower() == 'true'
    sampling = config['sampling'].lower()

    X, X_val, X_test = utils.get_datasets(dataset)
    num_samples, input_dim = X.shape
    delta = 1e-4 if dataset == 'lifesci' else 1 / num_samples

    shutil.copyfile(flow_path + 'flow_utils.py', 'analysis/flow_utils.py')
    from analysis import flow_utils

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, input_dim)

    iterations = sorted([int(d) for d in os.listdir(flow_path) if os.path.isdir(flow_path + d)])

    print('δ = {}'.format(delta))
    for composition in ['gdp', 'ma']:
        print('Composing in {}...'.format(composition))
        for iteration in iterations:
            epsilon = utils.get_epsilon(
	        private, composition, sampling, iteration,
	        noise_multiplier, num_samples, minibatch_size, delta,
            )
            params = pickle.load(open(flow_path + str(iteration) + '/params.pkl', 'rb'))
            likelihood = log_pdf(params, X_test).mean()
            print('ε: {:6g}\tLL: {:6g}'.format(epsilon, likelihood))
        print('-' * 30)
