import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


from .. import maf_utils as util
from .. import utils as repo_utils


path = 'datasets/mnist/mnist.pkl'


@repo_utils.constant_seed
def get_datasets():
    dataset = MNIST()
    return jnp.array(dataset.trn.x), jnp.array(dataset.val.x), jnp.array(dataset.tst.x)


class MNIST:
    """
    The MNIST dataset of handwritten digits.
    """

    alpha = 1.0e-6

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data, logit, dequantize, rng):

            x = self._dequantize(
                data[0], rng) if dequantize else data[0]  # dequantize pixels
            self.x = self._logit_transform(x) if logit else x  # logit
            self.labels = data[1]  # numeric labels
            self.y = util.one_hot_encode(self.labels,
                                         10)  # 1-hot encoded labels
            self.N = self.x.shape[0]  # number of datapoints
            self.x = self.x.astype('float32')
            self.y = self.y.astype('int')

        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return x + rng.rand(*x.shape) / 256.0

        @staticmethod
        def _logit_transform(x):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            return util.logit(MNIST.alpha + (1 - 2 * MNIST.alpha) * x)

    def __init__(self, logit=True, dequantize=True):

        # load dataset
        trn, val, tst = pickle.load(open(path, 'rb'), encoding='latin1')

        rng = np.random.RandomState(42)
        self.trn = self.Data(trn, logit, dequantize, rng)
        self.val = self.Data(val, logit, dequantize, rng)
        self.tst = self.Data(tst, logit, dequantize, rng)

        self.n_dims = self.trn.x.shape[1]
        self.n_labels = self.trn.y.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims))] * 2

    def show_pixel_histograms(self, split, pixel=None):
        """
        Shows the histogram of pixel values, or of a specific pixel if given.
        """

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        if pixel is None:
            data = data_split.x.flatten()

        else:
            row, col = pixel
            idx = row * self.image_size[0] + col
            data = data_split.x[:, idx]

        n_bins = int(np.sqrt(data_split.N))
        fig, ax = plt.subplots(1, 1)
        ax.hist(data, n_bins, normed=True)
        plt.show()

    def show_images(self, split):
        """
        Displays the images in a given split.
        :param split: string
        """

        # get split
        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        # display images
        util.disp_imdata(data_split.x, self.image_size, [6, 10])

        plt.show()
