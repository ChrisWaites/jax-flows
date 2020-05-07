import os
import sys

sys.path.insert(0, '../../')

from datetime import datetime
from tqdm import tqdm
import configparser
import itertools
import matplotlib.pyplot as plt
import numpy as onp
import numpy.random as npr
import pandas as pd
import pickle
import seaborn as sns
import shutil

from sklearn import datasets, preprocessing, mixture
from sklearn.decomposition import PCA

from jax import jit, grad, partial, random, tree_util, vmap, lax
from jax import numpy as np
from jax.experimental import optimizers, stax
from jax.nn.initializers import orthogonal, zeros, glorot_normal, normal

import flows
from datasets import *
from dp import compute_eps_poisson, compute_eps_uniform
import plotting


def shuffle(rng, arr):
    onp.random.seed(rng[1])
    arr = onp.asarray(arr).copy()
    onp.random.shuffle(arr)
    return np.array(arr)


def get_affine_coupling_net(input_shape, num_hidden=64, act=stax.Relu):
    return stax.serial(
        stax.Dense(num_hidden, W_init=orthogonal(), b_init=zeros),
        act,
        stax.Dense(num_hidden, W_init=orthogonal(), b_init=zeros),
        act,
        stax.Dense(input_shape[-1], W_init=orthogonal(), b_init=zeros),
    )


def get_affine_coupling_mask(input_shape):
    mask = onp.zeros(input_shape)
    mask[::2] = 1.0
    return mask


def main(config):
    print(dict(config))
    key = random.PRNGKey(0)

    b1 = float(config['b1'])
    b2 = float(config['b2'])
    delta = float(config['delta'])
    iterations = int(config['iterations'])
    dataset = config['dataset']
    target_epsilon = float(config['target_epsilon'])
    flow = config['flow']
    l2_norm_clip = float(config['l2_norm_clip'])
    log = str(config['log']).lower() == 'true'
    lr = float(config['lr'])
    microbatch_size = int(config['microbatch_size'])
    minibatch_size = int(config['minibatch_size'])
    num_blocks = int(config['num_blocks'])
    num_hidden = int(config['num_hidden'])
    noise_multiplier = float(config['noise_multiplier'])
    private = str(config['private']).lower() == 'true'
    weight_decay = float(config['weight_decay'])

    _, X, X_val, _ = {
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

    #scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler((-1., 1.))
    X = np.array(scaler.fit_transform(X))
    X_val = np.array(scaler.transform(X_val))

    input_shape = X.shape[1:]
    num_samples = X.shape[0]
    num_inputs = input_shape[-1]

    affine_coupling_scale = get_affine_coupling_net(input_shape, num_hidden, stax.Relu)
    affine_coupling_translate = get_affine_coupling_net(input_shape, num_hidden, stax.Tanh)
    affine_coupling_mask = get_affine_coupling_mask(input_shape)

    input_mask = flows.get_made_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
    hidden_mask = flows.get_made_mask(num_hidden, num_hidden, num_inputs)
    output_mask = flows.get_made_mask(num_hidden, num_inputs * 2, num_inputs, mask_type="output")

    joiner = flows.MaskedDense(num_hidden, input_mask)

    trunk = stax.serial(
        stax.Relu,
        flows.MaskedDense(num_hidden, hidden_mask),
        stax.Relu,
        flows.MaskedDense(num_inputs * 2, output_mask),
    )

    modules = []
    if flow == 'realnvp':
        mask = affine_coupling_mask
        for _ in range(num_blocks):
            modules += [
                flows.AffineCoupling(affine_coupling_scale, affine_coupling_translate, mask),
                flows.ActNorm(),
            ]
            mask = 1 - mask
    elif flow == 'glow':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(joiner, trunk, num_hidden),
                flows.ActNorm(),
                flows.InvertibleLinear(),
            ]
    elif flow == 'maf':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(joiner, trunk, num_hidden),
                flows.ActNorm(),
                flows.Reverse(),
            ]
    elif flow == 'neural-spline':
        for _ in range(num_blocks):
            modules += [
                flows.NeuralSplineCoupling(),
            ]
    elif flow == 'maf-glow':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(joiner, trunk, num_hidden),
                flows.ActNorm(),
                flows.InvertibleLinear(),
            ]
    else:
        raise Exception('Invalid flow: {}'.format(flow))

    bijection = flows.Serial(*tuple(modules))

    """
    gmm = mixture.GaussianMixture(6)
    gmm.fit(X)
    prior = flows.GMM(gmm.means_, gmm.covariances_, gmm.weights_)

    real_gmm_samples = gmm.sample(5000)[0]
    plt.hist2d(real_gmm_samples[:, 0], real_gmm_samples[:, 1], bins=100)
    plt.savefig('real.png')
    plt.clf()

    fake_gmm_samples = sample(random.PRNGKey(0), (), 5000)
    plt.hist2d(fake_gmm_samples[:, 0], fake_gmm_samples[:, 1], bins=100)
    plt.savefig('fake.png')
    plt.clf()
    """

    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)

    temp_key, key = random.split(key)
    params, log_pdf, sample = init_fun(temp_key, input_shape)

    def sched(i):
        return lr * np.minimum(1., (30000. / i) ** 2.)

    opt_init, opt_update, get_params = optimizers.adam(sched, b1, b2)
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


    # Plot training data
    if log:
        try:
            os.mkdir('out')
        except OSError as error:
            pass

        datetime_str = datetime_str = datetime.now().strftime('%b-%d-%Y_%I:%M:%S_%p')
        output_dir = 'out/' + datetime_str + '/'
        os.mkdir(output_dir)

        pickle.dump(dict(config), open(output_dir + 'config.pkl', 'wb'))

        shutil.copyfile('train.py', output_dir + 'train.py')

        if X.shape[1] == 2:
            plt.hist2d(X[:, 0], X[:, 1], bins=100, range=((-1.05, 1.05), (-1.05, 1.05)))
            plt.savefig(output_dir + 'samples.png')
            plt.clf()

        plotting.plot_marginals(X, output_dir)

    best_params, best_loss, best_loss_eps = None, None, None

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

        # Update progress bar
        if iteration % int(.005 * iterations) == 0:
            params = get_params(opt_state)
            loss_i = loss(params, X_val)
            epsilon_i = compute_eps_uniform(iteration, noise_multiplier, X.shape[0], minibatch_size, delta)

            if (private and epsilon_i >= target_epsilon) or iteration > 1000 and np.isnan(loss_i).any():
                return {'nll': (best_loss, 0.)}

            if best_loss is None or loss_i < best_loss:
                best_loss = loss_i

            pbar.set_description('NLL: {:.4f} Best NLL: {:.4f} ε: {:.4f}'.format(loss_i, best_loss if best_loss else 9999., epsilon_i))

        # Log to output directory
        if log and iteration % int(.05 * iterations) == 0:
            iteration_dir = output_dir + str(iteration) + '/'
            os.mkdir(iteration_dir)

            temp_key, key = random.split(key)
            X_syn = onp.asarray(sample(temp_key, params, num_samples))

            if X_syn.shape[1] == 2:
                plt.hist2d(X_syn[:, 0], X_syn[:, 1], bins=100, range=((-1.05, 1.05), (-1.05, 1.05)))
                plt.savefig(iteration_dir + 'samples.png')
                plt.clf()

            plotting.plot_marginals(X_syn, iteration_dir, overlay=X)

            pickle.dump(params, open(iteration_dir + 'params.pickle', 'wb'))

    return {'nll': (best_loss, 0.)}


if __name__ == '__main__':
    config_file = 'experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    print('Best validation loss: {}'.format(main(config)))
