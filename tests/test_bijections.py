import unittest

import jax.numpy as np
import numpy as onp
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Tanh
from jax.nn.initializers import orthogonal, zeros

import flows


def is_bijective(
    test, init_fun, inputs=random.uniform(random.PRNGKey(0), (20, 3), minval=-10.0, maxval=10.0), tol=1e-3
):
    input_shape = inputs.shape[1:]
    params, direct_fun, inverse_fun = init_fun(random.PRNGKey(0), input_shape)

    mapped_inputs = direct_fun(params, inputs)[0]
    reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]

    test.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))


def returns_correct_shape(
    test, init_fun, inputs=random.uniform(random.PRNGKey(0), (20, 3), minval=-10.0, maxval=10.0)
):
    input_shape = inputs.shape[1:]
    params, direct_fun, inverse_fun = init_fun(random.PRNGKey(0), input_shape)

    mapped_inputs, log_det_jacobian = direct_fun(params, inputs)
    test.assertTrue(inputs.shape == mapped_inputs.shape)
    test.assertTrue((inputs.shape[0], 1) == log_det_jacobian.shape)

    mapped_inputs, log_det_jacobian = inverse_fun(params, inputs)
    test.assertTrue(inputs.shape == mapped_inputs.shape)
    test.assertTrue((inputs.shape[0], 1) == log_det_jacobian.shape)


class Tests(unittest.TestCase):
    def test_shuffle(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.Shuffle())

    def test_reverse(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.Reverse())

    def test_affine_coupling(self):
        def net(input_shape, hidden_dim=64, act=Relu):
            return stax.serial(
                Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
                act,
                Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
                act,
                Dense(input_shape[-1], W_init=orthogonal(), b_init=zeros),
            )

        def mask(input_shape):
            mask = onp.zeros(input_shape)
            mask[::2] = 1.0
            return mask

        inputs = random.uniform(random.PRNGKey(0), (20, 3), minval=-10.0, maxval=10.0)

        init_fun = flows.AffineCoupling(
            net(input_shape=inputs.shape[1:], act=Relu),
            net(input_shape=inputs.shape[1:], act=Tanh),
            mask(input_shape=inputs.shape[1:]),
        )

        for test in (returns_correct_shape, is_bijective):
            test(self, init_fun, inputs)

    def test_made(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.MADE())

    def test_actnorm(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.ActNorm())

        # Test data-dependent initialization
        inputs = random.uniform(random.PRNGKey(0), (20, 3), minval=-10.0, maxval=10.0)

        init_fun = flows.serial(flows.ActNorm())
        params, direct_fun, inverse_fun = init_fun(random.PRNGKey(0), inputs.shape[1:], inputs=inputs)

        expected_weight, expected_bias = np.log(1.0 / (inputs.std(0) + 1e-12)), inputs.mean(0)

        self.assertTrue(np.array_equal(params[0][0], expected_weight))
        self.assertTrue(np.array_equal(params[0][1], expected_bias))

    def test_invertible_mm(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.InvertibleMM())

    def test_sigmoid(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.Sigmoid())

    def test_logit(self):
        inputs = random.uniform(random.PRNGKey(0), (20, 3))
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.Logit(), inputs)

    def test_serial(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.serial(flows.Shuffle(), flows.Shuffle()))

    def test_batchnorm(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.BatchNorm())
