import math

import jax.numpy as np
import numpy.random as npr


def Gaussian():
    def pdf(inputs):
        return (-0.5 * (inputs ** 2.0) - 0.5 * np.log(2 * math.pi)).sum(-1, keepdims=True)

    def sample(input_shape, num_samples=1):
        return npr.normal(0.0, 1.0, (num_samples,) + input_shape)

    return pdf, sample
