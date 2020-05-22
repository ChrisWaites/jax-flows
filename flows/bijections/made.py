import jax
import jax.numpy as np
from jax import random


def MaskedDense(mask):
    def init_fun(rng, input_shape):
        out_dim = mask.shape[-1]
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        bound = 1.0 / (input_shape[-1] ** 0.5)
        W = random.uniform(k1, (input_shape[-1], out_dim), minval=-bound, maxval=bound)
        b = random.uniform(k2, (out_dim,), minval=-bound, maxval=bound)
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return np.dot(inputs, W * mask) + b

    return init_fun, apply_fun


def MADE(transform):
    """An implementation of `MADE: Masked Autoencoder for Distribution Estimation`
    (https://arxiv.org/abs/1502.03509).

    Args:
        transform: maps inputs of dimension ``num_inputs`` to ``2 * num_inputs``

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        params, apply_fun = transform(rng, input_dim)

        def direct_fun(params, inputs, **kwargs):
            log_weight, bias = apply_fun(params, inputs).split(2, axis=1)
            outputs = (inputs - bias) * np.exp(-log_weight)
            log_det_jacobian = -log_weight.sum(-1)
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            outputs = np.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                log_weight, bias = apply_fun(params, outputs).split(2, axis=1)
                outputs = jax.ops.index_update(
                    outputs, jax.ops.index[:, i_col], inputs[:, i_col] * np.exp(log_weight[:, i_col]) + bias[:, i_col]
                )
            log_det_jacobian = -log_weight.sum(-1)
            return outputs, log_det_jacobian

        return params, direct_fun, inverse_fun

    return init_fun
