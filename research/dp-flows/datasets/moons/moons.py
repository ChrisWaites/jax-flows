import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets as ds

from .. import maf_utils as util
from .. import utils as dp_flow_utils


@dp_flow_utils.constant_seed
def get_datasets(val_prop=0.1):
    dataset = MOONS()
    return dataset.trn.x, dataset.val.x


class MOONS:
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        trn, val, tst = load_data()

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        util.plot_hist_marginals(data_split.x)
        plt.show()


def load_data():
    x = ds.make_moons(n_samples=30000, shuffle=True, noise=0.05)[0]
    return x[:24000], x[24000:27000], x[27000:]

