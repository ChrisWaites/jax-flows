from jax import jit, grad, partial, random, tree_util, vmap, lax
from jax import numpy as np
import configparser
import dp
import flow_utils
import flows
import matplotlib.pyplot as plt
import numpy as np
import pickle
import utils

dataset = 'lifesci'

paths = [
    ('out/' + dataset + '/flows/' + 'DP-NF-GDP' + '/',  'DP-NF (GDP)'),
    ('out/' + dataset + '/flows/' + 'DP-NF-MA' + '/',   'DP-NF (MA)'),
    ('out/' + dataset + '/gmm/' + 'DP-EM-MA' + '/',     'DP-EM (MA)'),
    ('out/' + dataset + '/gmm/' + 'DP-EM-zCDP' + '/',   'DP-EM (zCDP)'),
]

if __name__ == '__main__':
    # ------------------- NLL ------------------------

    plt.figure()

    for path, name in paths:
        epsilons = pickle.load(open(path + 'epsilons.pkl', 'rb'))
        losses = -np.array(pickle.load(open(path + 'val_losses.pkl', 'rb')))
        plt.plot(epsilons, losses, label=name)

    plt.xlabel('Privacy loss ε (δ = 1e-4)')
    plt.ylabel('Average log-likelihood')
    plt.legend()
    plt.grid(True)

    plt.savefig('lifesci-nll.png', dpi=1200)

    plt.clf()

    # ------------- Sample Quality -------------------

    from datasets import lifesci
    _, _, X = utils.get_datasets(dataset)
    input_shape = X.shape[1:]

    X_proj = lifesci.pca.transform(X)
    utils.plot_dist(X_proj, 'real_samples.png')

    for path, name in paths:
        if 'NF' in name:

            config_file = path + 'experiment.ini'
            config = configparser.ConfigParser()
            config.read(config_file)
            config = config['DEFAULT']

            flow = config['flow'].lower()
            num_blocks = int(config['num_blocks'])
            normalization = str(config['normalization']).lower() == 'true'
            num_hidden = int(config['num_hidden'])

            params = pickle.load(open(path + 'params.pkl', 'rb'))

            modules = flow_utils.get_modules(flow, num_blocks, input_shape, normalization, num_hidden)
            bijection = flows.Serial(*tuple(modules))
            prior = flows.Normal()

            key = random.PRNGKey(0)
            init_fun = flows.Flow(bijection, prior)
            temp_key, key = random.split(key)
            params, log_pdf, sample = init_fun(temp_key, input_shape)

            temp_key, key = random.split(key)
            X_syn = sample(temp_key, params, X.shape[0])
            X_syn_proj = lifesci.pca.transform(X_syn)

            utils.plot_dist(X_syn_proj, name + '_samples.png')
            # plot_marginals(X_syn, '', overlay=X):

