import math

import jax.numpy as np
from jax import random


def Normal():
    def init_fun(rng, input_shape):
        def log_pdf(params, inputs):
            inputs = inputs.reshape(inputs.shape[0], -1)
            return (-0.5 * (inputs ** 2.0) - 0.5 * np.log(2 * math.pi)).sum(-1, keepdims=True)

        def sample(rng, params, num_samples=1):
            return random.normal(rng, (num_samples,) + input_shape)

        return (), log_pdf, sample

    return init_fun


def Flow(transformation, prior=Normal()):
    def init_fun(rng, input_shape):
        transformation_rng, prior_rng = random.split(rng)

        params, direct_fun, inverse_fun = transformation(transformation_rng, input_shape)
        prior_params, prior_log_pdf, prior_sample = prior(prior_rng, input_shape)

        def log_pdf(params, inputs):
            u, log_det = direct_fun(params, inputs)
            log_probs = prior_log_pdf(prior_params, u)
            return (log_probs + log_det).sum(-1, keepdims=True)

        def sample(rng, params, num_samples=1):
            return inverse_fun(params, prior_sample(rng, prior_params, num_samples))[0]

        return params, log_pdf, sample

    return init_fun
