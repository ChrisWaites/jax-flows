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


def transform(rng, input_dim, output_dim, hidden_dim=64, act=stax.Relu):
    init_fun, apply_fun = stax.serial(
        stax.Dense(hidden_dim, weight_initializer, weight_initializer), act,
        stax.Dense(hidden_dim, weight_initializer, weight_initializer), act,
        stax.Dense(output_dim, weight_initializer, weight_initializer),
    )
    _, params = init_fun(rng, (input_dim,))
    return params, apply_fun


def masked_transform(rng, input_dim, output_dim, hidden_dim=64, act=stax.Relu):
    input_mask = flows.get_made_mask(input_dim, hidden_dim, input_dim, mask_type="input")
    hidden_mask = flows.get_made_mask(hidden_dim, hidden_dim, input_dim, mask_type=None)
    output_mask = flows.get_made_mask(hidden_dim, output_dim, input_dim, mask_type="output")

    init_fun, apply_fun = stax.serial(
        flows.MaskedDense(hidden_dim, input_mask), act,
        flows.MaskedDense(hidden_dim, hidden_mask), act,
        flows.MaskedDense(output_dim, output_mask),
    )
    _, params = init_fun(rng, (input_dim,))
    return params, apply_fun


def get_modules(flow, num_blocks, input_shape, normalization, num_hidden=64):
    num_inputs = input_shape[-1]

    input_mask = flows.get_made_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
    hidden_mask = flows.get_made_mask(num_hidden, num_hidden, num_inputs)
    output_mask = flows.get_made_mask(num_hidden, num_inputs * 2, num_inputs, mask_type="output")

    modules = []
    if flow == 'realnvp':
        for _ in range(num_blocks):
            modules += [
                flows.AffineCoupling(transform),
                flows.Reverse(),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
    elif flow == 'glow':
        for _ in range(num_blocks):
            modules += [
                flows.AffineCoupling(transform),
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
                flows.MADE(masked_transform),
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
                flows.MADE(masked_transform),
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
                flows.MADE(masked_transform),
                flows.InvertibleLinear(),
            ]
        modules += [
            flows.ActNorm(),
        ]
    else:
        raise Exception('Invalid flow: {}'.format(flow))

    return modules
