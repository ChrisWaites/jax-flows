import unittest

import jax.numpy as np
import numpy as onp
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Tanh
from jax.nn.initializers import orthogonal, zeros

import flows


class Tests(unittest.TestCase):
    def test_shuffle(self):
        num_examples, input_shape, tol = 100, (3,), 1e-4
        layer_rng, input_rng = random.split(random.PRNGKey(0))

        init_fun = flows.Shuffle()
        params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)

        inputs = random.uniform(input_rng, (num_examples,) + input_shape, minval=-10.0, maxval=10.0)
        mapped_inputs = direct_fun(params, inputs)[0]
        reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]

        self.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))

    def test_reverse(self):
        num_examples, input_shape, tol = 100, (3,), 1e-4
        layer_rng, input_rng = random.split(random.PRNGKey(0))

        init_fun = flows.Reverse()
        params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)

        inputs = random.uniform(input_rng, (num_examples,) + input_shape, minval=-10.0, maxval=10.0)
        mapped_inputs = direct_fun(params, inputs)[0]
        reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]

        self.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))

    def test_coupling(self):
        def net(rng, input_shape, hidden_dim=64, act=Relu):
            init_fun, apply_fun = stax.serial(
                Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
                act,
                Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
                act,
                Dense(input_shape[-1], W_init=orthogonal(), b_init=zeros),
            )
            _, params = init_fun(rng, input_shape)
            return (params, apply_fun)

        def mask(input_shape):
            mask = onp.zeros(input_shape)
            mask[::2] = 1.0
            return mask

        num_examples, input_shape, tol = 100, (3,), 1e-4
        layer_rng, input_rng, scale_rng, translate_rng = random.split(random.PRNGKey(0), 4)

        init_fun = flows.CouplingLayer(
            net(scale_rng, input_shape, act=Relu), net(translate_rng, input_shape, act=Tanh), mask(input_shape)
        )

        params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)

        inputs = random.uniform(input_rng, (num_examples,) + input_shape, minval=-10.0, maxval=10.0)
        mapped_inputs = direct_fun(params, inputs)[0]
        reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]

        self.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))

    def test_made(self):
        num_examples, input_shape, tol = 100, (3,), 1e-4
        layer_rng, input_rng = random.split(random.PRNGKey(0))

        init_fun = flows.MADE()
        params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)

        inputs = random.uniform(input_rng, (num_examples,) + input_shape, minval=-10.0, maxval=10.0)
        mapped_inputs = direct_fun(params, inputs)[0]
        reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]

        self.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))
