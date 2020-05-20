import jax
import jax.numpy as np
from jax import random
from jax.experimental import stax
from jax.nn.initializers import ones, normal


def MaskedDense(out_dim, mask):
    def init_fun(rng, input_shape):
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


def MADE(joiner, trunk, num_hidden):
    """An implementation of `MADE: Masked Autoencoder for Distribution Estimation`
    (https://arxiv.org/abs/1502.03509).

    Args:
        joiner: Maps inputs of dimension ``num_inputs`` to ``num_hidden``
        trunk: Maps inputs of dimension ``num_hidden`` to ``2 * num_inputs``
        num_hidden: The hidden dimension of choice

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_shape, **kwargs):
        joiner_rng, trunk_rng = random.split(rng)

        joiner_init_fun, joiner_apply_fun = joiner
        _, joiner_params = joiner_init_fun(joiner_rng, input_shape)

        trunk_init_fun, trunk_apply_fun = trunk
        _, trunk_params = trunk_init_fun(trunk_rng, (num_hidden,))

        def direct_fun(params, inputs, **kwargs):
            joiner_params, trunk_params = params
            h = joiner_apply_fun(joiner_params, inputs)
            m, a = trunk_apply_fun(trunk_params, h).split(2, 1)
            u = (inputs - m) * np.exp(-a)
            log_det_jacobian = -a.sum(-1)
            return u, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            joiner_params, trunk_params = params
            x = np.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = joiner_apply_fun(joiner_params, x)
                m, a = trunk_apply_fun(trunk_params, h).split(2, 1)
                x = jax.ops.index_update(
                    x, jax.ops.index[:, i_col], inputs[:, i_col] * np.exp(a[:, i_col]) + m[:, i_col]
                )
            log_det_jacobian = -a.sum(-1)
            return x, log_det_jacobian

        return (joiner_params, trunk_params), direct_fun, inverse_fun
    return init_fun


def MADESplit(s_joiner, s_trunk, t_joiner, t_trunk, num_hidden):
    """An implementation of `MADE: Masked Autoencoder for Distribution Estimation`
    (https://arxiv.org/abs/1502.03509).

    Args:
        joiner: Maps inputs of dimension ``num_inputs`` to ``num_hidden``
        trunk: Maps inputs of dimension ``num_hidden`` to ``2 * num_inputs``
        num_hidden: The hidden dimension of choice

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_shape, **kwargs):
        s_joiner_rng, rng = random.split(rng)
        s_joiner_init_fun, s_joiner_apply_fun = s_joiner
        _, s_joiner_params = s_joiner_init_fun(s_joiner_rng, input_shape)

        s_trunk_rng, rng = random.split(rng)
        s_trunk_init_fun, s_trunk_apply_fun = s_trunk
        _, s_trunk_params = s_trunk_init_fun(s_trunk_rng, (num_hidden,))

        t_joiner_rng, rng = random.split(rng)
        t_joiner_init_fun, t_joiner_apply_fun = t_joiner
        _, t_joiner_params = t_joiner_init_fun(t_joiner_rng, input_shape)

        t_trunk_rng, rng = random.split(rng)
        t_trunk_init_fun, t_trunk_apply_fun = t_trunk
        _, t_trunk_params = t_trunk_init_fun(t_trunk_rng, (num_hidden,))

        def direct_fun(params, inputs, **kwargs):
            s_joiner_params, s_trunk_params, t_joiner_params, t_trunk_params = params

            h = s_joiner_apply_fun(s_joiner_params, inputs)
            m = s_trunk_apply_fun(s_trunk_params, h)

            h = t_joiner_apply_fun(t_joiner_params, inputs)
            a = t_trunk_apply_fun(t_trunk_params, h)

            a = np.tanh(a)
            u = (inputs - m) * np.exp(-a)

            log_det_jacobian = -a.sum(-1)
            return u, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            s_joiner_params, s_trunk_params, t_joiner_params, t_trunk_params = params
            x = np.zeros_like(inputs)

            for i_col in range(inputs.shape[1]):
                h = s_joiner_apply_fun(s_joiner_params, x)
                m = s_trunk_apply_fun(s_trunk_params, h)

                h = t_joiner_apply_fun(t_joiner_params, inputs)
                a = t_trunk_apply_fun(t_trunk_params, h)

                a = np.tanh(a)

                x = jax.ops.index_update(
                    x, jax.ops.index[:, i_col], inputs[:, i_col] * np.exp(a[:, i_col]) + m[:, i_col]
                )

            log_det_jacobian = -a.sum(-1)
            return x, log_det_jacobian

        return (s_joiner_params, s_trunk_params, t_joiner_params, t_trunk_params), direct_fun, inverse_fun

    return init_fun
