import unittest
from jax import random
from jax.nn.initializers import orthogonal, zeros
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Tanh
from flax import transformations
import jax.numpy as np
import numpy as onp


class TestTransformations(unittest.TestCase):
    def test_shuffle(self):
        num_examples, input_shape, tol = 100, (3,), 1e-4
        layer_rng, input_rng = random.split(random.PRNGKey(0))

        init_fun = transformations.Shuffle()
        params, normalizing_fun, generative_fun = init_fun(layer_rng, input_shape)

        inputs = random.uniform(input_rng, (num_examples,) + input_shape, minval=-10., maxval=10.)
        mapped_inputs = normalizing_fun(params, inputs)[0]
        reconstructed_inputs = generative_fun(params, mapped_inputs)[0]

        self.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))


    def test_reverse(self):
        num_examples, input_shape, tol = 100, (3,), 1e-4
        layer_rng, input_rng = random.split(random.PRNGKey(0))

        init_fun = transformations.Reverse()
        params, normalizing_fun, generative_fun = init_fun(layer_rng, input_shape)

        inputs = random.uniform(input_rng, (num_examples,) + input_shape, minval=-10., maxval=10.)
        mapped_inputs = normalizing_fun(params, inputs)[0]
        reconstructed_inputs = generative_fun(params, mapped_inputs)[0]

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

        init_fun = transformations.CouplingLayer(
            net(scale_rng, input_shape, act=Relu),
            net(translate_rng, input_shape, act=Tanh),
            mask(input_shape)
        )

        params, normalizing_fun, generative_fun = init_fun(layer_rng, input_shape)

        inputs = random.uniform(input_rng, (num_examples,) + input_shape, minval=-10., maxval=10.)
        mapped_inputs = normalizing_fun(params, inputs)[0]
        reconstructed_inputs = generative_fun(params, mapped_inputs)[0]

        self.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))


    def test_made(self):
        num_examples, input_shape, tol = 100, (3,), 1e-4
        layer_rng, input_rng = random.split(random.PRNGKey(0))

        init_fun = transformations.MADE()
        params, normalizing_fun, generative_fun = init_fun(layer_rng, input_shape)

        inputs = random.uniform(input_rng, (num_examples,) + input_shape, minval=-10., maxval=10.)
        mapped_inputs = normalizing_fun(params, inputs)[0]
        reconstructed_inputs = generative_fun(params, mapped_inputs)[0]

        self.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))

