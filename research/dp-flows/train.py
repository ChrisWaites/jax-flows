import os
import sys

sys.path.insert(0, '../../')

from datetime import datetime
from tqdm import tqdm
import configparser
import numpy as onp
import pickle
import shutil
from sklearn import preprocessing

from jax import jit, grad, partial, random, tree_util, vmap, lax
from jax import numpy as np
from jax.experimental import optimizers

import dp
import flow_utils
import flows
import utils


def main(config):
    print(dict(config))
    key = random.PRNGKey(0)

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
    optimizer = config['optimizer']
    private = str(config['private']).lower() == 'true'
    weight_decay = float(config['weight_decay'])

    # Create dataset
    _, X, X_val, _ = utils.get_datasets(dataset)

    scaler = preprocessing.MinMaxScaler((-1., 1.)) # preprocessing.StandardScaler()
    X = np.array(scaler.fit_transform(X))
    X_val = np.array(scaler.transform(X_val))

    input_shape = X.shape[1:]
    num_samples = X.shape[0]
    num_inputs = input_shape[-1]

    # Create flow
    modules = flow_utils.get_modules(flow, num_blocks, input_shape, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)

    temp_key, key = random.split(key)
    params, log_pdf, sample = init_fun(temp_key, input_shape)

    # Create optimizer
    # sched = lambda i: lr * np.minimum(1., (i / 10000.) ** 2.)
    # sched = lambda i: lr * np.minimum(1., (50000. / i) ** 2.)
    sched = lambda i: lr * (0.99995) ** i
    if optimizer.lower() == 'adagrad':
        opt_init, opt_update, get_params = optimizers.adagrad(sched)
    elif optimizer.lower() == 'adam':
        opt_init, opt_update, get_params = optimizers.adam(sched)
    elif optimizer.lower() == 'momentum':
        opt_init, opt_update, get_params = optimizers.momentum(sched)
    elif optimizer.lower() == 'sgd':
        opt_init, opt_update, get_params = optimizers.sgd(sched)
    else:
        raise Exception('Invalid optimizer: {}'.format(optimizer))
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
        utils.make_dir('out')
        datetime_str = datetime.now().strftime('%b-%d-%Y_%I:%M:%S_%p')
        output_dir = 'out/' + datetime_str + '/'
        utils.make_dir(output_dir)

        # Log config, experiment file, and plot distribution
        pickle.dump(dict(config), open(output_dir + 'config.pkl', 'wb'))
        shutil.copyfile('train.py', output_dir + 'train.py')
        if X.shape[1] == 2:
            utils.plot_dist(X, output_dir + 'real.png')
        utils.plot_marginals(X, output_dir)

    best_params, best_loss = None, None
    train_losses, val_losses = [], []
    pbar = tqdm(range(iterations))
    for iteration in pbar:
        # Calculate current epoch from iteration, etc.
        epoch = iteration // (X.shape[0] // minibatch_size)
        batch_index = iteration % (X.shape[0] // minibatch_size)
        batch_index_start = batch_index * minibatch_size

        # Shuffle dataset and calculate batch
        if batch_index == 0:
            temp_key, key = random.split(key)
            X = random.permutation(temp_key, X)
        batch = X[batch_index_start:batch_index_start+minibatch_size]

        # Perform update from batch
        temp_key, key = random.split(key)
        if private:
            opt_state = private_update(temp_key, iteration, opt_state, batch)
        else:
            opt_state = update(temp_key, iteration, opt_state, batch)

        # Update progress bar
        if iteration % int(.005 * iterations) == 0:
            params = get_params(opt_state)
            train_loss = loss(params, X)
            val_loss = loss(params, X_val)

            epsilon_i = dp.compute_eps_uniform(iteration, noise_multiplier, X.shape[0], minibatch_size, delta)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (private and epsilon_i >= target_epsilon) or iteration > 1000 and np.isnan(train_loss).any():
                if log:
                    pickle.dump(best_params, open(output_dir + 'params.pickle', 'wb'))
                    utils.plot_loss(train_losses, val_losses, output_dir + 'loss.png')
                print('Encountered NaN, exiting.')
                return {'nll': (best_loss, 0.)}

            if best_loss is None or val_loss < best_loss:
                bset_params = params
                best_loss = val_loss

            pbar_text = 'Train NLL: {:.4f} Val NLL: {:.4f} Best NLL: {:.4f} ε: {:.4f}'.format(
                train_loss,
                val_loss,
                best_loss if best_loss else 9999.,
                epsilon_i
            )
            pbar.set_description(pbar_text)

        # Log params and plots to output directory
        if log and iteration % int(.1 * iterations) == 0:
            iteration_dir = output_dir + str(iteration) + '/'
            utils.make_dir(iteration_dir)

            temp_key, key = random.split(key)
            X_syn = onp.asarray(sample(temp_key, params, num_samples))

            pickle.dump(params, open(iteration_dir + 'params.pickle', 'wb'))
            if X_syn.shape[1] == 2:
                utils.plot_dist(X_syn, iteration_dir + 'synthetic.png')
            utils.plot_marginals(X_syn, iteration_dir, overlay=X)

    if log:
        pickle.dump(best_params, open(output_dir + 'best_params.pickle', 'wb'))
        utils.plot_loss(train_losses, val_losses, output_dir + 'loss.png')
    return {'nll': (best_loss, 0.)}


if __name__ == '__main__':
    config_file = 'experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    print('Best validation loss: {}'.format(main(config)))
