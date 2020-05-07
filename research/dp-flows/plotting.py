import seaborn as sns
import matplotlib.pyplot as plt

def plot_marginals(X, path, overlay=None):
    for dim in range(X.shape[1]):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist(X[:, dim], color='red', alpha=0.5, bins=30, range=(-1.05, 1.05))
        if not (overlay is None):
            ax.hist(overlay[:, dim], color='blue', alpha=0.2, bins=30, range=(-1.05, 1.05))
        plt.savefig(path + str(dim) + '.png')
        plt.clf()
