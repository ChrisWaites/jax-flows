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
    normalization = str(config['normalization']).lower() == 'true'
    optimizer = config['optimizer'].lower()
    private = str(config['private']).lower() == 'true'
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

    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    params, log_pdf, sample = init_fun(temp_key, input_shape)

    # Create optimizer
    # sched = lambda i: lr * np.minimum(1., (i / 10000.) ** 2.)
    # sched = lambda i: lr * np.minimum(1., (50000. / i) ** 2.)
    # sched = lambda i: lr * (0.99995 ** i)
    sched = lr
    opt_init, opt_update, get_params = utils.get_optimizer(optimizer, sched, b1, b2)
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
        # Create experiment directory
        datetime_str = datetime.now().strftime('%b-%d-%Y_%I:%M:%S_%p')

        output_dir = ''
        for ext in ['out', dataset, 'flows', datetime_str]:
            output_dir += ext + '/'
            utils.make_dir(output_dir)

        # Log files and plot real distribution
        utils.dump_obj(dict(config), output_dir + 'config.pkl')
        shutil.copyfile('train.py', output_dir + 'train.py')
        shutil.copyfile('flow_utils.py', output_dir + 'flow_utils.py')

        if X.shape[1] == 2:
            utils.plot_dist(X, output_dir + 'real.png')

        if X.shape[1] <= 16:
            utils.plot_marginals(X, output_dir)

    best_params, best_loss = None, None
    train_losses, val_losses, epsilons = [], [], []
    pbar = tqdm(range(iterations))
    for iteration in pbar:
        # Calculate epoch from iteration
        epoch = iteration // (X.shape[0] // minibatch_size)
        batch_index = iteration % (X.shape[0] // minibatch_size)
        batch_index_start = batch_index * minibatch_size

        # Regular batching
        #if batch_index == 0:
        #    temp_key, key = random.split(key)
        #    X = random.permutation(temp_key, X)
        #batch = X[batch_index_start:batch_index_start+minibatch_size]

        # Poisson subsampling
        #temp_key, key = random.split(key)
        #whether = random.uniform(temp_key, (num_samples,)) < (minibatch_size / num_samples)
        #batch = X[whether]

        # Uniform subsampling
        temp_key, key = random.split(key)
        X = random.permutation(temp_key, X)
        batch = X[:minibatch_size]

        if batch.shape[0] == 0:
            continue

        # Perform model update
        temp_key, key = random.split(key)
        if private:
            opt_state = private_update(temp_key, iteration, opt_state, batch)
        else:
            opt_state = update(temp_key, iteration, opt_state, batch)

        # Update progress bar
        if iteration % int(.005 * iterations) == 0:
            # Calculate privacy loss
            try:
                epsilon = dp.compute_eps_uniform(
                    iteration, noise_multiplier,
                    X.shape[0], minibatch_size, delta
                )
            except:
                epsilon = dp.epsilon(
                    X.shape[0], minibatch_size,
                    noise_multiplier, iteration, delta,
                )

            # Calculate losses
            params = get_params(opt_state)
            train_loss = loss(params, X)
            val_loss = loss(params, X_val)

            # Exit if privacy limit reached or NaN occurs
            if (private and epsilon >= target_epsilon) or iteration > 5000 and np.isnan(train_loss).any():
                if log:
                    utils.log(
                        key, best_params, sample, X,
                        output_dir + str(best_loss).replace('.', '_') + '/',
                        train_losses, val_losses, epsilons,
                    )
                return {'nll': (best_loss, 0.)}

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            epsilons.append(epsilon)

            # Update best model thus far
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                best_params = params

            # Update progress bar
            pbar_text = 'Train NLL: {:.4f} Val NLL: {:.4f} Best NLL: {:.4f} ε: {:.4f}'.format(
                train_loss,
                val_loss,
                best_loss if best_loss else 99999.,
                epsilon,
            )

            pbar.set_description(pbar_text)

        # Log params and plots to output directory
        if log and iteration % int(.05 * iterations) == 0:
            utils.log(
                key, params, sample, X,
                output_dir + str(iteration) + '/',
            )

    if log:
        utils.log(
            key, best_params, sample, X,
            output_dir + str(best_loss).replace('.', '_') + '/',
            train_losses, val_losses, epsilons,
        )
    return {'nll': (best_loss, 0.)}


if __name__ == '__main__':
    config_file = 'experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    print('Best validation loss: {}'.format(main(config)))
