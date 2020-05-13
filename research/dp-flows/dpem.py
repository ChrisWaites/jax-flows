from scipy.io import loadmat
import glob


def get_gmm_params(path):
    mat = loadmat(path)

    means = mat['model']['cpd'][0][0][0][0][0].transpose()
    covariances = mat['model']['cpd'][0][0][0][0][1].transpose()
    weights = mat['model']['mixWeight'][0][0][0]
    epsilon = float(path.split('_')[3][8:])

    return epsilon, means, covariances, weights

def get_all_gmm_params(base_path):
    results = []
    for path in glob.glob(base_path + "*.mat"):
        if 'epsilon' in path:
            results.append(get_gmm_params(path))
    return results

