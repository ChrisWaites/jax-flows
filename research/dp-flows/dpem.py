from scipy.io import loadmat

path = '/Users/chriswaites/Documents/projects/dpem_code/matfiles/'
path += 'lifesci_G=lap_0_epsilon=4_delta=0.0001_comp=4.mat'


def get_gmm_params(path=path):
    mat = loadmat(path)

    means = mat['model']['cpd'][0][0][0][0][0].transpose()
    covariances = mat['model']['cpd'][0][0][0][0][1].transpose()
    weights = mat['model']['mixWeight'][0][0][0]

    epsilon = float(path.split('_')[4][8:])

    return means, covariances, weights, eps

