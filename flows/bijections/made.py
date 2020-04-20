import jax
import jax.numpy as np
from jax import random
from jax.experimental import stax
from jax.nn.initializers import glorot_normal, normal


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

            log_det_jacobian = -a.sum(-1, keepdims=True)

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

            log_det_jacobian = -a.sum(-1, keepdims=True)

            return x, log_det_jacobian

        return (joiner_params, trunk_params), direct_fun, inverse_fun

    return init_fun
