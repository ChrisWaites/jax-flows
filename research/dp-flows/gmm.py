import sys

sys.path.insert(0, '../../')

import configparser
from sklearn import preprocessing, mixture
from jax import numpy as np
import numpy as np
import utils


def main(config):
    n_components = 30
    max_iter = 10

    dataset = config['dataset']
    _, X, X_val, _ = utils.get_datasets(dataset)

    #scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler((-1., 1.))
    X = scaler.fit_transform(X)
    X_val = scaler.transform(X_val)

    GMM = mixture.GaussianMixture(n_components=n_components, max_iter=max_iter)
    GMM.fit(X)

    nll = -GMM.score_samples(X_val).mean()
    X_syn = GMM.sample(X.shape[0])[0]

    output_dir = ''
    for ext in ['out', dataset, 'gmm', str(n_components)]:
        output_dir += ext + '/'
        utils.make_dir(output_dir)

    if X.shape[1] == 2:
        utils.plot_dist(X, output_dir + 'real.png')
        utils.plot_dist(X_syn, output_dir + 'synthetic.png')

    utils.plot_marginals(X_syn, output_dir, overlay=X)

    return {'nll': (nll, 0.)}


if __name__ == '__main__':
    config_file = 'experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    print('Best validation loss: {}'.format(main(config)))
