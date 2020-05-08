import math

import jax.numpy as np
import jax.scipy.stats.multivariate_normal as mvn
from jax import random
from jax.scipy.special import logsumexp


def Normal():
    """
    Returns:
        A function mapping ``(rng, input_shape)`` to a ``(params, log_pdf, sample)`` triplet.
    """

    def init_fun(rng, input_shape):
        def log_pdf(params, inputs):
            inputs = inputs.reshape(inputs.shape[0], -1)
            return (-0.5 * (inputs ** 2.0) - 0.5 * np.log(2.0 * math.pi)).sum(-1)

        def sample(rng, params, num_samples=1):
            return random.normal(rng, (num_samples,) + input_shape)

        return (), log_pdf, sample
    return init_fun


def GMM(means, covariances, weights):
    def init_fun(rng, input_shape):
        def log_pdf(params, inputs):
            cluster_lls = []
            for log_weight, mean, cov in zip(np.log(weights), means, covariances):
                cluster_lls.append(log_weight + mvn.logpdf(inputs, mean, cov))
            return logsumexp(np.vstack(cluster_lls), axis=0)

        def sample(rng, params, num_samples=1):
            cluster_samples = []
            for mean, cov in zip(means, covariances):
                rng, temp_rng = random.split(rng)
                cluster_sample = random.multivariate_normal(temp_rng, mean, cov, (num_samples,))
                cluster_samples.append(cluster_sample)
            samples = np.dstack(cluster_samples)
            idx = random.categorical(rng, weights, shape=(num_samples, 1, 1))
            return np.squeeze(np.take_along_axis(samples, idx, -1))

        return (), log_pdf, sample
    return init_fun


def Flow(transformation, prior=Normal()):
    """
    Args:
        transformation: a function mapping ``(rng, input_shape)`` to a
            ``(params, direct_fun, inverse_fun)`` triplet
        prior: a function mapping ``(rng, input_shape)`` to a
            ``(params, log_pdf, sample)`` triplet

    Returns:
        A function mapping ``(rng, input_shape)`` to a ``(params, log_pdf, sample)`` triplet.

    Examples:
        >>> import flows
        >>> input_shape, rng = (3,), random.PRNGKey(0)
        >>> transformation = flows.Serial(
        ...     flows.Reverse(),
        ...     flows.Reverse()
        ... )
        >>> init_fun = flows.Flow(transformation, Normal())
        >>> params, log_pdf, sample = init_fun(rng, input_shape)
    """

    def init_fun(rng, input_shape):
        transformation_rng, prior_rng = random.split(rng)

        params, direct_fun, inverse_fun = transformation(transformation_rng, input_shape)
        prior_params, prior_log_pdf, prior_sample = prior(prior_rng, input_shape)

        def log_pdf(params, inputs):
            u, log_det = direct_fun(params, inputs)
            log_probs = prior_log_pdf(prior_params, u)
            return log_probs + log_det

        def sample(rng, params, num_samples=1):
            prior_samples = prior_sample(rng, prior_params, num_samples)
            return inverse_fun(params, prior_samples)[0]

        return params, log_pdf, sample
    return init_fun
