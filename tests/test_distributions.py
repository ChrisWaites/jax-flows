import unittest

import jax.numpy as np
import numpy as onp
from jax import random
from jax.experimental import stax
from jax.nn.initializers import orthogonal, zeros
from sklearn import mixture, datasets
from jax.scipy.stats import multivariate_normal

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
        inputs = random.uniform(random.PRNGKey(0), (20, 2), minval=-3.0, maxval=3.0)
        input_shape = inputs.shape[1:]
        init_key, sample_key = random.split(random.PRNGKey(0))

        init_fun = flows.Normal()
        params, log_pdf, sample = init_fun(init_key, input_shape)
        log_pdfs = log_pdf(params, inputs)

        mean = np.zeros(input_shape)
        covariance = np.eye(np.prod(input_shape))
        true_log_pdfs = multivariate_normal.logpdf(inputs, mean, covariance)

        self.assertTrue(np.allclose(log_pdfs, true_log_pdfs))

        for test in (returns_correct_shape,):
            test(self, flows.Normal())

    def test_gmm(self):
        inputs = datasets.make_blobs()[0]
        input_shape = inputs.shape[1:]
        init_key, sample_key = random.split(random.PRNGKey(0))

        gmm = mixture.GaussianMixture(3)
        gmm.fit(inputs)
        init_fun = flows.GMM(gmm.means_, gmm.covariances_, gmm.weights_)

        params, log_pdf, sample = init_fun(init_key, input_shape)
        log_pdfs = log_pdf(params, inputs)

        self.assertTrue(np.allclose(log_pdfs, gmm.score_samples(inputs)))

        for test in (returns_correct_shape,):
            test(self, init_fun, inputs)

    def test_flow(self):
        for test in (returns_correct_shape,):
            test(self, flows.Flow(flows.Reverse(), flows.Normal()))
