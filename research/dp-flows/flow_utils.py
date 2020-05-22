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
        flows.MaskedDense(masks[0]), act,
        flows.MaskedDense(masks[1]), act,
        flows.MaskedDense(masks[2].tile(2)),
    )
    _, params = init_fun(rng, (input_dim,))
    return params, apply_fun


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
                flows.MADE(masked_transform),
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
