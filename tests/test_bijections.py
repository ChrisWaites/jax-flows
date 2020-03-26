import unittest

import jax.numpy as np
import numpy as onp
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Tanh
from jax.nn.initializers import orthogonal, zeros

import flows


def is_bijective(test, init_fun, num_examples=100, input_shape=(3,), minval=-10.0, maxval=10.0, tol=1e-3):
    layer_rng, input_rng = random.split(random.PRNGKey(0))

    params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)

    inputs = random.uniform(input_rng, (num_examples,) + input_shape, minval=minval, maxval=maxval)
    mapped_inputs = direct_fun(params, inputs)[0]
    reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]

    test.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))


class Tests(unittest.TestCase):
    def test_shuffle(self):
        is_bijective(self, flows.Shuffle())

    def test_reverse(self):
        is_bijective(self, flows.Reverse())

    def test_affine_coupling(self):
        def net(input_shape=(3,), hidden_dim=64, act=Relu):
            return stax.serial(
                Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
                act,
                Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
                act,
                Dense(input_shape[-1], W_init=orthogonal(), b_init=zeros),
            )

        def mask(input_shape=(3,)):
            mask = onp.zeros(input_shape)
            mask[::2] = 1.0
            return mask

        input_shape = (3,)

        init_fun = flows.AffineCoupling(
            net(input_shape=input_shape, act=Relu),
            net(input_shape=input_shape, act=Tanh),
            mask(input_shape=input_shape),
        )

        is_bijective(self, init_fun, input_shape=input_shape)

    def test_made(self):
        is_bijective(self, flows.MADE())

    def test_actnorm(self):
        is_bijective(self, flows.ActNorm())

    def test_sigmoid(self):
        is_bijective(self, flows.Sigmoid())

    def test_logit(self):
        is_bijective(self, flows.Logit(), minval=0.0, maxval=1.0)
