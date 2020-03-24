import jax.numpy as np
import math
import numpy.random as npr


def Gaussian():
  def pdf(inputs):
    return (-.5 * (inputs ** 2.) - .5 * np.log(2 * math.pi)).sum(-1, keepdims=True)

  def sample(input_shape, num_samples=1):
    return npr.normal(0., 1., (num_samples,) + input_shape)

  return pdf, sample

