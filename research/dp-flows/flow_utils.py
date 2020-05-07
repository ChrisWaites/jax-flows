import sys

sys.path.insert(0, '../../')

from jax.experimental import stax
from jax.nn.initializers import orthogonal, zeros
import flows
import numpy as onp


def get_affine_coupling_net(input_shape, num_hidden=64, act=stax.Relu):
    return stax.serial(
        stax.Dense(num_hidden, W_init=orthogonal(), b_init=zeros),
        act,
        stax.Dense(num_hidden, W_init=orthogonal(), b_init=zeros),
        act,
        stax.Dense(input_shape[-1], W_init=orthogonal(), b_init=zeros),
    )


def get_affine_coupling_mask(input_shape):
    mask = onp.zeros(input_shape)
    mask[::2] = 1.0
    return mask


def get_modules(flow, num_blocks, input_shape, num_hidden=64):
    num_inputs = input_shape[-1]

    affine_coupling_scale = get_affine_coupling_net(input_shape, num_hidden, stax.Relu)
    affine_coupling_translate = get_affine_coupling_net(input_shape, num_hidden, stax.Tanh)
    affine_coupling_mask = get_affine_coupling_mask(input_shape)

    input_mask = flows.get_made_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
    hidden_mask = flows.get_made_mask(num_hidden, num_hidden, num_inputs)
    output_mask = flows.get_made_mask(num_hidden, num_inputs * 2, num_inputs, mask_type="output")

    made_joiner = flows.MaskedDense(num_hidden, input_mask)
    made_trunk = stax.serial(
        stax.Relu,
        flows.MaskedDense(num_hidden, hidden_mask),
        stax.Relu,
        flows.MaskedDense(num_inputs * 2, output_mask),
    )

    modules = []
    if flow == 'realnvp':
        mask = affine_coupling_mask
        for _ in range(num_blocks):
            modules += [
                flows.AffineCoupling(affine_coupling_scale, affine_coupling_translate, mask),
                flows.ActNorm(),
            ]
            mask = 1 - mask
    elif flow == 'glow':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(made_joiner, made_trunk, num_hidden),
                flows.ActNorm(),
                flows.InvertibleLinear(),
            ]
    elif flow == 'maf':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(made_joiner, made_trunk, num_hidden),
                flows.ActNorm(),
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
                flows.MADE(made_joiner, made_trunk, num_hidden),
                flows.ActNorm(),
                flows.InvertibleLinear(),
            ]
    else:
        raise Exception('Invalid flow: {}'.format(flow))

    return modules
