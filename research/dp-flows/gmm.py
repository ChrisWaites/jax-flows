import sys

sys.path.insert(0, '../../')

import configparser
from sklearn import preprocessing, mixture
from sklearn.mixture import _gaussian_mixture
from jax import numpy as np
from jax import random
import numpy as np

import utils
import dpem
import flows


def main(config):
    n_components = int(config['n_components'])
    dataset = config['dataset'].lower()
    split = int(config['split'])
    max_iter = int(config['max_iter'])
    log = str(config['log']).lower() == 'true'
    private = str(config['private']).lower() == 'true'

    _, X, X_val, _ = utils.get_datasets(dataset)

    GMM = mixture.GaussianMixture(n_components=n_components, max_iter=max_iter)

    if private:
        """
        init_fun = flows.GMM(*dpem.get_gmm_params())
        params, log_pdf, sample = init_fun(random.PRNGKey(0), X.shape[1:])
        nll = -log_pdf(params, X_val).mean()
        X_syn = sample(random.PRNGKey(0), params, X.shape[0])
        """
        GMM.means_, GMM.covariances_, GMM.weights_, epsilon = dpem.get_gmm_params()
        GMM.precisions_cholesky_ = _gaussian_mixture._compute_precision_cholesky(GMM.covariances_, 'full')
        nll = -GMM.score_samples(X_val).mean()
        X_syn = GMM.sample(X.shape[0])[0]
    else:
        GMM.fit(X)
        nll = -GMM.score_samples(X_val).mean()
        X_syn = GMM.sample(X.shape[0])[0]

    output_dir = ''
    for ext in ['out', dataset, 'gmm', 'private' if private else 'nonprivate', str(n_components)]:
        output_dir += ext + '/'
        utils.make_dir(output_dir)

    if X.shape[1] == 2:
        utils.plot_dist(X, output_dir + 'real.png')
        utils.plot_dist(X_syn, output_dir + 'synthetic.png')

    utils.plot_marginals(X_syn, output_dir, overlay=X)

    return {'nll': (nll, 0.)}


if __name__ == '__main__':
    config_file = 'gmm_experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    print('Validation loss: {}'.format(main(config)))
