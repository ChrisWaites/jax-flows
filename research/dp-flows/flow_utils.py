import sys

sys.path.insert(0, '../../')

from jax import random
from jax.experimental import stax
import flows
import jax.numpy as np


def MNISTAffineCoupling(transform):
    def init_fun(rng, input_dim, **kwargs):
        cutoff = input_dim // 2
        params, apply_fun = transform(rng, cutoff, 2 * (input_dim - cutoff))

        def direct_fun(params, inputs, **kwargs):
            lower, upper = inputs[:, :cutoff], inputs[:, cutoff:]

            log_weight, bias = apply_fun(params, upper).split(2, axis=1)
            lower = lower * np.exp(log_weight) + bias

            outputs = np.concatenate([lower, upper], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            lower, upper = inputs[:, :cutoff], inputs[:, cutoff:]

            log_weight, bias = apply_fun(params, upper).split(2, axis=1)
            lower = (lower - bias) * np.exp(-log_weight)

            outputs = np.concatenate([lower, upper], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs, log_det_jacobian

        return params, direct_fun, inverse_fun

    return init_fun


def weight_initializer(key, shape, dtype=np.float32):
    bound = 1.0 / (shape[0] ** 0.5)
    return random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def get_transform(hidden_dim=64, act=stax.Relu):
    def transform(rng, input_dim, output_dim):
        init_fun, apply_fun = stax.serial(
            stax.Dense(hidden_dim, weight_initializer, weight_initializer),
            act,
            stax.Dense(hidden_dim, weight_initializer, weight_initializer),
            act,
            stax.Dense(output_dim, weight_initializer, weight_initializer),
        )
        _, params = init_fun(rng, (input_dim,))
        return params, apply_fun
    return transform


def get_nice_transform(hidden_dim=64):
    def transform(rng, input_dim, output_dim):
        params, apply_fun = get_transform(hidden_dim=hidden_dim)(rng, input_dim, output_dim)
        def new_apply_fun(params, inputs):
            log_weight, bias = apply_fun(params, inputs).split(2, axis=1)
            result =  np.concatenate((np.zeros_like(log_weight), bias), axis=1)
            return result
        return params, new_apply_fun
    return transform


def get_transform(hidden_dim=64, act=stax.Relu):
    def transform(rng, input_dim, output_dim):
        init_fun, apply_fun = stax.serial(
            stax.Dense(hidden_dim, weight_initializer, weight_initializer),
            act,
            stax.Dense(hidden_dim, weight_initializer, weight_initializer),
            act,
            stax.Dense(output_dim, weight_initializer, weight_initializer),
        )
        _, params = init_fun(rng, (input_dim,))
        return params, apply_fun
    return transform


def get_conv_transform(hidden_dim=64, act=stax.Relu):
    def Reshape():
        def init_fun(rng, input_shape):
            return (-1, 28, 14, 1), () # round(input_shape[1] ** 0.5), round(input_shape[1] ** 0.5), 1), ()

        def apply_fun(params, inputs, **kwargs):
            return inputs.reshape((inputs.shape[0], 28, 14, 1)) # round(input_shape[1] ** 0.5), round(input_shape[1] ** 0.5), 1)

        return init_fun, apply_fun

    def transform(rng, input_dim, output_dim):
        init_fun, apply_fun = stax.serial(
            Reshape(),
            stax.Conv(8, filter_shape=(3, 3), W_init=weight_initializer, b_init=weight_initializer),
            act,
            stax.Conv(16, filter_shape=(3, 3), W_init=weight_initializer, b_init=weight_initializer),
            act,
            stax.Flatten,
            stax.Dense(output_dim, W_init=weight_initializer, b_init=weight_initializer),
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


def get_masked_transform(hidden_dim=64, act=stax.Relu):
    def masked_transform(rng, input_dim):
        masks = get_masks(input_dim, hidden_dim)
        init_fun, apply_fun = stax.serial(
            flows.MaskedDense(masks[0]), act,
            flows.MaskedDense(masks[1]), act,
            flows.MaskedDense(masks[2].tile(2)),
        )
        _, params = init_fun(rng, (input_dim,))
        return params, apply_fun
    return masked_transform


def get_modules(flow, num_blocks, normalization, hidden_dim=64):
    modules = []
    if flow == 'realnvp':
        for _ in range(num_blocks):
            modules += [
                flows.AffineCoupling(get_transform(hidden_dim)),
                flows.Reverse(),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
    elif flow == 'realnvp-conv':
        for _ in range(num_blocks):
            modules += [
                MNISTAffineCoupling(get_conv_transform(hidden_dim)),
                flows.Reverse(),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
    elif flow == 'nice':
        for _ in range(num_blocks):
            modules += [
                flows.AffineCoupling(get_nice_transform(hidden_dim)),
                flows.Reverse(),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
    elif flow == 'glow':
        for _ in range(num_blocks):
            modules += [
                flows.AffineCoupling(get_transform(hidden_dim)),
                flows.InvertibleLinear(),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
    elif flow == 'maf':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(get_masked_transform(hidden_dim)),
                flows.Reverse(),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
    elif flow == 'neural-spline':
        for _ in range(num_blocks):
            modules += [
                flows.NeuralSplineCoupling(),
            ]
    elif flow == 'maf-glow':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(get_masked_transform(hidden_dim)),
                flows.InvertibleLinear(),
            ]
            if normalization:
                modules += [
                    flows.ActNorm(),
                ]
    elif flow == 'custom':
        for _ in range(num_blocks):
            modules += [
                flows.MADE(get_masked_transform(hidden_dim)),
                flows.InvertibleLinear(),
            ]
        modules += [
            flows.ActNorm(),
        ]
    else:
        raise Exception('Invalid flow: {}'.format(flow))

    return modules
