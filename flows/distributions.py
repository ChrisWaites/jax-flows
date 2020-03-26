import math

import jax.numpy as np
from jax import random


def Normal():
    """
    Returns:
        A function mapping `(rng, input_shape)` to a `(params, log_pdf, sample)` triplet
    """

    def init_fun(rng, input_shape):
        def log_pdf(params, inputs):
            inputs = inputs.reshape(inputs.shape[0], -1)
            return (-0.5 * (inputs ** 2.0) - 0.5 * np.log(2 * math.pi)).sum(-1, keepdims=True)

        def sample(rng, params, num_samples=1):
            return random.normal(rng, (num_samples,) + input_shape)

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
        A function mapping ``(rng, input_shape)`` to a
            ``(params, log_pdf, sample)`` triplet

    Examples:
        >>> import flows
        >>> input_shape, rng = (3,), random.PRNGKey(0)
        >>> transformation = flows.serial(
        ...     flows.MADE(),
        ...     flows.Reverse(),
        ...     flows.MADE(),
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
            return (log_probs + log_det).sum(-1, keepdims=True)

        def sample(rng, params, num_samples=1):
            prior_samples = prior_sample(rng, prior_params, num_samples)
            return inverse_fun(params, prior_samples)[0]

        return params, log_pdf, sample

    return init_fun
