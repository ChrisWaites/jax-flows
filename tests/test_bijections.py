import unittest

import jax.numpy as np
from jax import random
from jax.experimental import stax

import flows


def is_bijective(
    test, init_fun, inputs=random.uniform(random.PRNGKey(0), (20, 4), minval=-10.0, maxval=10.0), tol=1e-3
):
    input_dim = inputs.shape[1]
    params, direct_fun, inverse_fun = init_fun(random.PRNGKey(0), input_dim)

    mapped_inputs = direct_fun(params, inputs)[0]
    reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]

    test.assertTrue(np.allclose(inputs, reconstructed_inputs, atol=tol))


def returns_correct_shape(
    test, init_fun, inputs=random.uniform(random.PRNGKey(0), (20, 4), minval=-10.0, maxval=10.0)
):
    input_dim = inputs.shape[1]
    params, direct_fun, inverse_fun = init_fun(random.PRNGKey(0), input_dim)

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
                stax.Dense(hidden_dim), act, stax.Dense(hidden_dim), act, stax.Dense(output_dim),
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
        def get_masks(input_dim, hidden_dim=64, num_hidden=1):
            masks = []
            input_degrees = np.arange(input_dim)
            degrees = [input_degrees]

            for n_h in range(num_hidden + 1):
                degrees += [np.arange(hidden_dim) % (input_dim - 1)]
            degrees += [input_degrees % input_dim - 1]

            for (d0, d1) in zip(degrees[:-1], degrees[1:]):
                masks += [np.transpose(np.expand_dims(d1, -1) >= np.expand_dims(d0, 0)).astype(np.float32)]
            return masks

        def masked_transform(rng, input_dim):
            masks = get_masks(input_dim, hidden_dim=64, num_hidden=1)
            act = stax.Relu
            init_fun, apply_fun = stax.serial(
                flows.MaskedDense(masks[0]),
                act,
                flows.MaskedDense(masks[1]),
                act,
                flows.MaskedDense(masks[2].tile(2)),
            )
            _, params = init_fun(rng, (input_dim,))
            return params, apply_fun

        for test in (returns_correct_shape, is_bijective):
            test(self, flows.MADE(masked_transform))

    def test_actnorm(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.ActNorm())

        # Test data-dependent initialization
        inputs = random.uniform(random.PRNGKey(0), (20, 3), minval=-10.0, maxval=10.0)
        input_dim = inputs.shape[1]

        init_fun = flows.Serial(flows.ActNorm())
        params, direct_fun, inverse_fun = init_fun(random.PRNGKey(0), inputs.shape[1:], init_inputs=inputs)
        mapped_inputs, _ = direct_fun(params, inputs)

        self.assertFalse((np.abs(mapped_inputs.mean(0)) > 1e6).any())
        self.assertTrue(np.allclose(np.ones(input_dim), mapped_inputs.std(0)))

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
