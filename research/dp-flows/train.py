import os
import sys

sys.path.insert(0, '../../')

from datetime import datetime
from tqdm import tqdm
import configparser
import numpy as onp
import pickle
import shutil

from jax import jit, grad, partial, random, tree_util, vmap, lax
from jax import numpy as np

import dp
import flow_utils
import flows
import utils


def main(config):
    print(dict(config))
    key = random.PRNGKey(0)

    b1 = float(config['b1'])
    b2 = float(config['b2'])
    composition = config['composition'].lower()
    dataset = config['dataset']
    delta = float(config['delta'])
    dirname = config['dirname']
    flow = config['flow']
    iterations = int(config['iterations'])
    l2_norm_clip = float(config['l2_norm_clip'])
    log_params = None if config['log_params'].lower() == 'none' else int(config['log_params'])
    log_performance = int(config['log_performance'])
    lr = float(config['lr'])
    lr_schedule = config['lr_schedule'].lower()
    microbatch_size = int(config['microbatch_size'])
    minibatch_size = int(config['minibatch_size'])
    noise_multiplier = float(config['noise_multiplier'])
    normalization = str(config['normalization']).lower() == 'true'
    num_blocks = int(config['num_blocks'])
    num_hidden = int(config['num_hidden'])
    optimizer = config['optimizer'].lower()
    private = str(config['private']).lower() == 'true'
    sampling = config['sampling'].lower()
    target_epsilon = float(config['target_epsilon'])
    weight_decay = float(config['weight_decay'])

    # Create dataset
    _, X, X_val = utils.get_datasets(dataset)

    input_shape = X.shape[1:]
    num_samples = X.shape[0]
    num_inputs = input_shape[-1]

    # Create flow
    modules = flow_utils.get_modules(flow, num_blocks, input_shape, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()

    """
    from sklearn import mixture
    gmm = mixture.GaussianMixture(n_components=10)
    gmm.fit(X)
    prior = flows.GMM(gmm.means_, gmm.covariances_, gmm.weights_)
    """

    init_fun = flows.Flow(lambda key, shape: bijection(key, shape, init_inputs=X), prior)
    temp_key, key = random.split(key)
    params, log_pdf, sample = init_fun(temp_key, input_shape)

    # Create optimizer
    scheduler = utils.get_scheduler(lr, lr_schedule)
    opt_init, opt_update, get_params = utils.get_optimizer(optimizer, scheduler, b1, b2)
    opt_state = opt_init(params)

    def l2(pytree):
        leaves, _ = tree_util.tree_flatten(pytree)
        return np.sqrt(sum(np.vdot(x, x) for x in leaves))

    def loss(params, inputs):
        # return np.nan_to_num(-log_pdf(params, inputs)).clip(-1e6, 1e6).mean() + weight_decay * l2(params)
        return -log_pdf(params, inputs).mean() + weight_decay * l2(params)

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
    if log_params:
        # Create experiment directory
        # dirname = datetime.now().strftime('%b-%d-%Y_%I:%M:%S_%p')

        output_dir = ''
        for ext in ['out', dataset, 'flows', dirname]:
            output_dir += ext + '/'
            utils.make_dir(output_dir)

        # Log files and plot real distribution
        shutil.copyfile('experiment.ini', output_dir + 'experiment.ini')
        shutil.copyfile('train.py', output_dir + 'train.py')
        shutil.copyfile('flow_utils.py', output_dir + 'flow_utils.py')

        """
        if X.shape[1] == 2:
            utils.plot_dist(X, output_dir + 'real.png')

        if X.shape[1] <= 16:
            utils.plot_marginals(X, output_dir)
        """

    best_params, best_loss = None, None
    train_losses, val_losses, epsilons = [], [], []
    pbar = tqdm(range(1, iterations + 1))
    for iteration in pbar:
        batch, X = utils.get_batch(sampling, key, X, minibatch_size, iteration)

        if batch.shape[0] == 0:
            continue

        # Perform model update
        temp_key, key = random.split(key)
        if private:
            opt_state = private_update(temp_key, iteration, opt_state, batch)
        else:
            opt_state = update(temp_key, iteration, opt_state, batch)

        # Update progress bar
        if iteration % log_performance == 0:
            # Calculate privacy loss
            epsilon = utils.get_epsilon(
                sampling, composition, private,
                iteration, noise_multiplier,
                num_samples, minibatch_size, delta,
            )

            # Calculate losses
            params = get_params(opt_state)
            train_loss = loss(params, X)
            val_loss = loss(params, X_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            epsilons.append(epsilon)

            # Update best model thus far
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                best_params = params

            if log_params:
                utils.log(key, best_params, sample, X, output_dir, train_losses, val_losses, epsilons)

            # Update progress bar
            pbar_text = 'Train: {:.3f} Val: {:.3f} Best: {:.3f} ε: {:.3f}'.format(
                train_loss,
                val_loss,
                best_loss,
                epsilon,
            )

            pbar.set_description(pbar_text)

        # Log params and plots to output directory
        if log_params and iteration % log_params == 0:
            utils.log(key, params, sample, X, output_dir + str(iteration) + '/')

    if log_params:
        utils.log(key, best_params, sample, X, output_dir, train_losses, val_losses, epsilons)
    return {'nll': (best_loss, 0.)}


if __name__ == '__main__':
    config_file = 'experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    print('Best validation loss: {}'.format(main(config)))
