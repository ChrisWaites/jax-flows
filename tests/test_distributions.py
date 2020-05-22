import unittest

import jax.numpy as np
from jax import random
from jax.scipy.stats import multivariate_normal
from sklearn import datasets, mixture

import flows


def returns_correct_shape(
    test, init_fun, inputs=random.uniform(random.PRNGKey(0), (20, 4), minval=-10.0, maxval=10.0)
):
    num_inputs, input_dim = inputs.shape

    init_key, sample_key = random.split(random.PRNGKey(0))
    params, log_pdf, sample = init_fun(init_key, input_dim)

    log_pdfs = log_pdf(params, inputs)
    samples = sample(sample_key, params, num_inputs)

    test.assertTrue(log_pdfs.shape == (num_inputs,))
    test.assertTrue(samples.shape == inputs.shape)


class Tests(unittest.TestCase):
    def test_normal(self):
        inputs = random.uniform(random.PRNGKey(0), (20, 2), minval=-3.0, maxval=3.0)
        input_dim = inputs.shape[1]
        init_key, sample_key = random.split(random.PRNGKey(0))

        init_fun = flows.Normal()
        params, log_pdf, sample = init_fun(init_key, input_dim)
        log_pdfs = log_pdf(params, inputs)

        mean = np.zeros(input_dim)
        covariance = np.eye(input_dim)
        true_log_pdfs = multivariate_normal.logpdf(inputs, mean, covariance)

        self.assertTrue(np.allclose(log_pdfs, true_log_pdfs))

        for test in (returns_correct_shape,):
            test(self, flows.Normal())

    def test_gmm(self):
        inputs = datasets.make_blobs()[0]
        input_dim = inputs.shape[1]
        init_key, sample_key = random.split(random.PRNGKey(0))

        gmm = mixture.GaussianMixture(3)
        gmm.fit(inputs)
        init_fun = flows.GMM(gmm.means_, gmm.covariances_, gmm.weights_)

        params, log_pdf, sample = init_fun(init_key, input_dim)
        log_pdfs = log_pdf(params, inputs)

        self.assertTrue(np.allclose(log_pdfs, gmm.score_samples(inputs)))

        for test in (returns_correct_shape,):
            test(self, init_fun, inputs)

    def test_flow(self):
        for test in (returns_correct_shape,):
            test(self, flows.Flow(flows.Reverse(), flows.Normal()))
