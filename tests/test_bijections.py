import unittest

import jax.numpy as np
import numpy as onp
from jax import random
from jax.experimental import stax
from jax.nn.initializers import orthogonal, zeros

import flows


def is_bijective(
    test, init_fun, inputs=random.uniform(random.PRNGKey(0), (20, 4), minval=-10.0, maxval=10.0), tol=1e-3
):
    input_shape = inputs.shape[1:]
    params, direct_fun, inverse_fun = init_fun(random.PRNGKey(0), input_shape)

    mapped_inputs = direct_fun(params, inputs)[0]
    reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]

    test.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))


def returns_correct_shape(
    test, init_fun, inputs=random.uniform(random.PRNGKey(0), (20, 4), minval=-10.0, maxval=10.0)
):
    input_shape = inputs.shape[1:]
    params, direct_fun, inverse_fun = init_fun(random.PRNGKey(0), input_shape)

    mapped_inputs, log_det_jacobian = direct_fun(params, inputs)
    test.assertTrue(inputs.shape == mapped_inputs.shape)
    test.assertTrue((inputs.shape[0],) == log_det_jacobian.shape)

    mapped_inputs, log_det_jacobian = inverse_fun(params, inputs)
    test.assertTrue(inputs.shape == mapped_inputs.shape)
    test.assertTrue((inputs.shape[0],) == log_det_jacobian.shape)


class Tests(unittest.TestCase):
    def test_shuffle(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.Shuffle())

    def test_reverse(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.Reverse())

    def test_affine_coupling(self):
        def transform(rng, input_dim, output_dim, hidden_dim=64, act=stax.Relu):
            init_fun, apply_fun = stax.serial(
                stax.Dense(hidden_dim), act,
                stax.Dense(hidden_dim), act,
                stax.Dense(output_dim),
            )
            _, params = init_fun(rng, (input_dim,))
            return params, apply_fun

        inputs = random.uniform(random.PRNGKey(0), (20, 5), minval=-10.0, maxval=10.0)

        init_fun = flows.AffineCoupling(transform)
        for test in (returns_correct_shape, is_bijective):
            test(self, init_fun, inputs)

        init_fun = flows.AffineCouplingSplit(transform, transform)
        for test in (returns_correct_shape, is_bijective):
            test(self, init_fun, inputs)

    def test_made(self):
        inputs = random.uniform(random.PRNGKey(0), (20, 4), minval=-10.0, maxval=10.0)

        input_shape = inputs.shape[1:]
        hidden_dim = 64
        input_dim = input_shape[-1]

        def autoencoder(rng, input_dim, output_dim, hidden_dim=64):
            input_rng, hidden_rng, output_rng = random.split(rng, 3)
            input_mask = flows.get_made_mask(input_rng, input_dim, hidden_dim, input_dim, mask_type="input")
            hidden_mask = flows.get_made_mask(hidden_rng, hidden_dim, hidden_dim, input_dim, mask_type=None)
            output_mask = flows.get_made_mask(output_rng, hidden_dim, output_dim, input_dim, mask_type="output")

            init_fun, apply_fun = stax.serial(
                flows.MaskedDense(hidden_dim, input_mask), stax.Relu,
                flows.MaskedDense(hidden_dim, hidden_mask), stax.Relu,
                flows.MaskedDense(output_dim, output_mask),
            )
            _, params = init_fun(rng, (input_dim,))
            return params, apply_fun

        for test in (returns_correct_shape, is_bijective):
            test(self, flows.MADE(autoencoder))

    def test_actnorm(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.ActNorm())

        # Test data-dependent initialization
        inputs = random.uniform(random.PRNGKey(0), (20, 3), minval=-10.0, maxval=10.0)
        input_shape = inputs.shape[1:]

        init_fun = flows.Serial(flows.ActNorm())
        params, direct_fun, inverse_fun = init_fun(random.PRNGKey(0), inputs.shape[1:], init_inputs=inputs)
        expected_weight, expected_bias = np.log(1.0 / (inputs.std(0) + 1e-12)), inputs.mean(0)
        mapped_inputs, _ = direct_fun(params, inputs)

        self.assertFalse((np.abs(mapped_inputs.mean(0)) > 1e6).any())
        self.assertTrue(np.allclose(np.ones(input_shape), mapped_inputs.std(0)))

    def test_invertible_linear(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.InvertibleLinear())
               
    def test_fixed_invertible_linear(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.FixedInvertibleLinear())

    def test_sigmoid(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.Sigmoid())

    def test_logit(self):
        inputs = random.uniform(random.PRNGKey(0), (20, 3))
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.Logit(), inputs)

    def test_serial(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.Serial(flows.Shuffle(), flows.Shuffle()))

    def test_batchnorm(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.BatchNorm())

    def test_neural_spline(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.NeuralSplineCoupling())
        """
        init_fun = flows.NeuralSplineCoupling()

        inputs = random.uniform(random.PRNGKey(0), (20, 2), minval=-10.0, maxval=10.0)
        tol = 1e-3
        input_shape = inputs.shape[1:]

        params, direct_fun, inverse_fun = init_fun(random.PRNGKey(0), input_shape)

        # -----------------
        mapped_inputs = direct_fun(params, inputs)[0]
        reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]

        self.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))
        # -----------------
        mapped_inputs, log_det_jacobian = direct_fun(params, inputs)
        self.assertTrue(inputs.shape == mapped_inputs.shape)
        test.assertTrue((inputs.shape[0], 1) == log_det_jacobian.shape)

        mapped_inputs, log_det_jacobian = inverse_fun(params, inputs)
        test.assertTrue(inputs.shape == mapped_inputs.shape)
        test.assertTrue((inputs.shape[0], 1) == log_det_jacobian.shape)
        """
