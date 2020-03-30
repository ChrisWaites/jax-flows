import unittest

import jax.numpy as np
import numpy as onp
from jax import random
from jax.experimental import stax
from jax.nn.initializers import glorot_normal, normal, orthogonal, zeros

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
        def get_affine_coupling_net(input_shape, hidden_dim=64, act=stax.Relu):
            return stax.serial(
                stax.Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
                act,
                stax.Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
                act,
                stax.Dense(input_shape[-1], W_init=orthogonal(), b_init=zeros),
            )

        def get_affine_coupling_mask(input_shape):
            mask = onp.zeros(input_shape)
            mask[::2] = 1.0
            return mask

        inputs = random.uniform(random.PRNGKey(0), (20, 3), minval=-10.0, maxval=10.0)

        init_fun = flows.AffineCoupling(
            get_affine_coupling_net(input_shape=inputs.shape[1:], act=stax.Relu),
            get_affine_coupling_net(input_shape=inputs.shape[1:], act=stax.Tanh),
            get_affine_coupling_mask(input_shape=inputs.shape[1:]),
        )

        for test in (returns_correct_shape, is_bijective):
            test(self, init_fun, inputs)

    def test_made(self):
        def MaskedDense(out_dim, mask, W_init=glorot_normal(), b_init=normal()):
            init_fun, _ = stax.Dense(out_dim, W_init, b_init)

            def apply_fun(params, inputs, **kwargs):
                W, b = params
                return np.dot(inputs, W * mask) + b

            return init_fun, apply_fun

        def get_made_mask(in_features, out_features, in_flow_features, mask_type=None):
            if mask_type == "input":
                in_degrees = np.arange(in_features) % in_flow_features
            else:
                in_degrees = np.arange(in_features) % (in_flow_features - 1)

            if mask_type == "output":
                out_degrees = np.arange(out_features) % in_flow_features - 1
            else:
                out_degrees = np.arange(out_features) % (in_flow_features - 1)

            mask = np.expand_dims(out_degrees, -1) >= np.expand_dims(in_degrees, 0)
            return np.transpose(mask).astype(np.float32)

        inputs = random.uniform(random.PRNGKey(0), (20, 3), minval=-10.0, maxval=10.0)

        input_shape = inputs.shape[1:]
        num_hidden = 64
        num_inputs = input_shape[-1]

        input_mask = get_made_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
        hidden_mask = get_made_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_made_mask(num_hidden, num_inputs * 2, num_inputs, mask_type="output")

        joiner = MaskedDense(num_hidden, input_mask)
        trunk = stax.serial(
            stax.Relu, MaskedDense(num_hidden, hidden_mask), stax.Relu, MaskedDense(num_inputs * 2, output_mask)
        )

        for test in (returns_correct_shape, is_bijective):
            test(self, flows.MADE(joiner, trunk, num_hidden))

    def test_actnorm(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.ActNorm())

        # Test data-dependent initialization
        inputs = random.uniform(random.PRNGKey(0), (20, 3), minval=-10.0, maxval=10.0)

        init_fun = flows.Serial(flows.ActNorm())
        params, direct_fun, inverse_fun = init_fun(random.PRNGKey(0), inputs.shape[1:], inputs=inputs)

        expected_weight, expected_bias = np.log(1.0 / (inputs.std(0) + 1e-12)), inputs.mean(0)

        self.assertTrue(np.array_equal(params[0][0], expected_weight))
        self.assertTrue(np.array_equal(params[0][1], expected_bias))

    def test_invertible_linear(self):
        for test in (returns_correct_shape, is_bijective):
            test(self, flows.InvertibleLinear())

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
