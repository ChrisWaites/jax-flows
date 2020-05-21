import sys

sys.path.insert(0, '../../')

from jax import random
from jax.experimental import stax
from jax.nn.initializers import orthogonal, zeros
import flows
import jax.numpy as np
import numpy as onp


def weight_initializer(key, shape, dtype=np.float32):
    bound = 1.0 / (shape[0] ** 0.5)
    return random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def get_transform(hidden_dim=64):
    def transform(rng, input_dim, output_dim, hidden_dim=64, act=stax.Relu):
        init_fun, apply_fun = stax.serial(
            stax.Dense(hidden_dim, weight_initializer, weight_initializer), act,
            stax.Dense(hidden_dim, weight_initializer, weight_initializer), act,
            stax.Dense(output_dim, weight_initializer, weight_initializer),
        )
        _, params = init_fun(rng, (input_dim,))
        return params, apply_fun
    return transform


def get_masked_transform(hidden_dim=64):
    def masked_transform(rng, input_dim, output_dim, act=stax.Relu):
        input_rng, hidden_rng, output_rng, rng = random.split(rng, 4)
        input_mask = flows.get_made_mask(input_rng, input_dim, hidden_dim, input_dim, mask_type="input")
        hidden_mask = flows.get_made_mask(hidden_rng, hidden_dim, hidden_dim, input_dim, mask_type=None)
        output_mask = flows.get_made_mask(output_rng, hidden_dim, output_dim, input_dim, mask_type="output")
        init_fun, apply_fun = stax.serial(
            flows.MaskedDense(hidden_dim, input_mask), act,
            flows.MaskedDense(hidden_dim, hidden_mask), act,
            flows.MaskedDense(output_dim, output_mask),
        )
        _, params = init_fun(rng, (input_dim,))
        return params, apply_fun
    return masked_transform


def get_modules(flow, num_blocks, input_shape, normalization, num_hidden=64):
    num_inputs = input_shape[-1]

    modules = []
    if flow == 'realnvp':
        for _ in range(num_blocks):
            modules += [
                flows.AffineCoupling(get_transform(num_hidden)),
                flows.Reverse(),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
    elif flow == 'glow':
        for _ in range(num_blocks):
            modules += [
                flows.AffineCoupling(get_transform(num_hidden)),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
            modules += [
                flows.InvertibleLinear(),
            ]
    elif flow == 'maf':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(get_masked_transform(num_hidden)),
            ]
            if normalization:
                modules += [
                    flows.BatchNorm(),
                ]
            modules += [
                flows.Reverse(),
            ]
    elif flow == 'neural-spline':
        for _ in range(num_blocks):
            modules += [
                flows.NeuralSplineCoupling(),
            ]
    elif flow == 'maf-glow':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(get_masked_transform(num_hidden)),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
            modules += [
                flows.InvertibleLinear(),
            ]
    elif flow == 'custom':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(get_masked_transform(num_hidden)),
                flows.InvertibleLinear(),
            ]
        modules += [
            flows.ActNorm(),
        ]
    else:
        raise Exception('Invalid flow: {}'.format(flow))

    return modules
