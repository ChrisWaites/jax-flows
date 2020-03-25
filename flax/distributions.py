import math

import jax.numpy as np
from jax import random


def Normal():
    def logpdf(inputs):
        inputs = inputs.reshape(inputs.shape[0], -1)
        return (-0.5 * (inputs ** 2.0) - 0.5 * np.log(2 * math.pi)).sum(-1, keepdims=True)

    def sample(rng, input_shape, num_samples=1):
        return random.normal(rng, (num_samples,) + input_shape)

    return logpdf, sample
