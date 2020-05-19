import sys

sys.path.insert(0, '../../')

from jax.experimental import stax
from jax.nn.initializers import orthogonal, zeros
import flows
import numpy as onp
import jax.numpy as np


def weight_initializer(key, shape, dtype=np.float32):
    bound = 1.0 / (shape[0] ** 0.5)
    return random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def get_affine_coupling_net(input_shape, num_hidden=64, act=stax.Relu):
    return stax.serial(
        stax.Dense(num_hidden, orthogonal(), zeros),
        act,
        stax.Dense(num_hidden, orthogonal(), zeros),
        act,
        stax.Dense(input_shape[-1], orthogonal(), zeros),
    )


def get_affine_coupling_mask(input_shape):
    mask = onp.zeros(input_shape)
    mask[::2] = 1.0
    return mask


def get_modules(flow, num_blocks, input_shape, normalization, num_hidden=64):
    num_inputs = input_shape[-1]

    affine_coupling_scale = get_affine_coupling_net(input_shape, num_hidden, stax.Relu)
    affine_coupling_translate = get_affine_coupling_net(input_shape, num_hidden, stax.Tanh)
    affine_coupling_mask = get_affine_coupling_mask(input_shape)

    input_mask = flows.get_made_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
    hidden_mask = flows.get_made_mask(num_hidden, num_hidden, num_inputs)
    output_mask = flows.get_made_mask(num_hidden, num_inputs * 2, num_inputs, mask_type="output")

    made_act = stax.Relu
    made_joiner = flows.MaskedDense(num_hidden, input_mask)
    made_trunk = stax.serial(
        made_act,
        flows.MaskedDense(num_hidden, hidden_mask),
        made_act,
        flows.MaskedDense(num_inputs * 2, output_mask),
    )

    modules = []
    if flow == 'realnvp':
        for _ in range(num_blocks):
            modules += [
                flows.AffineCoupling(
                    affine_coupling_scale,
                    affine_coupling_translate,
                    affine_coupling_mask,
                ),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
            affine_coupling_mask = 1 - affine_coupling_mask
    elif flow == 'glow':
        for _ in range(num_blocks):
            modules += [
                flows.AffineCoupling(
                    affine_coupling_scale,
                    affine_coupling_translate,
                    affine_coupling_mask,
                ),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
            modules += [
                flows.InvertibleLinear(),
            ]
            affine_coupling_mask = 1 - affine_coupling_mask
    elif flow == 'maf':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(
                    made_joiner,
                    made_trunk,
                    num_hidden,
                ),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
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
                flows.MADE(
                    made_joiner,
                    made_trunk,
                    num_hidden,
                ),
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
                flows.MADE(
                    made_joiner,
                    made_trunk,
                    num_hidden,
                ),
                flows.InvertibleLinear(),
            ]
        modules += [
            flows.ActNorm(),
        ]
    else:
        raise Exception('Invalid flow: {}'.format(flow))

    return modules
