import unittest

import jax.numpy as np
import numpy as onp
from jax import random
from jax.experimental import stax
from jax.nn.initializers import orthogonal, zeros
from sklearn import mixture, datasets

import flows


def returns_correct_shape(
    test, init_fun, inputs=random.uniform(random.PRNGKey(0), (20, 4), minval=-10.0, maxval=10.0)
):
    input_shape = inputs.shape[1:]
    init_key, sample_key = random.split(random.PRNGKey(0))
    params, log_pdf, sample = init_fun(init_key, input_shape)

    log_pdfs = log_pdf(params, inputs)
    samples = sample(sample_key, params, inputs.shape[0])

    test.assertTrue(log_pdfs.shape == (inputs.shape[0],))
    test.assertTrue(samples.shape == inputs.shape)


class Tests(unittest.TestCase):
    def test_normal(self):
        for test in (returns_correct_shape,):
            test(self, flows.Normal())

    def test_gmm(self):
        X = datasets.make_blobs()[0]

        gmm = mixture.GaussianMixture(3)
        gmm.fit(X)
        init_fun = flows.GMM(gmm.means_, gmm.covariances_, gmm.weights_)

        input_shape = X.shape[1:]
        init_key, sample_key = random.split(random.PRNGKey(0))
        params, log_pdf, sample = init_fun(init_key, input_shape)
        log_pdfs = log_pdf(params, X)

        self.assertTrue(np.allclose(log_pdfs, gmm.score_samples(X)))

        for test in (returns_correct_shape,):
            test(self, init_fun, X)

    def test_flow(self):
        for test in (returns_correct_shape,):
            test(self, flows.Flow(flows.Reverse(), flows.Normal()))
