import jax
import jax.numpy as np
import numpy as onp
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu
from jax.nn.initializers import glorot_normal, normal


def Shuffle():
    def init_fun(rng, input_shape):
        perm = random.shuffle(rng, np.arange(onp.prod(input_shape))).reshape(input_shape)
        inv_perm = np.argsort(perm)

        def direct_fun(params, inputs, **kwargs):
            return inputs[:, perm], np.zeros((inputs.shape[0], 1))

        def inverse_fun(params, inputs, **kwargs):
            return inputs[:, inv_perm], np.zeros((inputs.shape[0], 1))

        return (), direct_fun, inverse_fun

    return init_fun


def Reverse():
    def init_fun(rng, input_shape):
        perm = np.array(np.arange(onp.prod(input_shape))[::-1]).reshape(input_shape)
        inv_perm = np.argsort(perm)

        def direct_fun(params, inputs, **kwargs):
            return inputs[:, perm], np.zeros((inputs.shape[0], 1))

        def inverse_fun(params, inputs, **kwargs):
            return inputs[:, inv_perm], np.zeros((inputs.shape[0], 1))

        return (), direct_fun, inverse_fun

    return init_fun


def CouplingLayer(scale, translate, mask):
    """
    Args:
        *scale: A trainable scaling function, i.e. a (params, apply_fun) pair
        *translate: A trainable translation function, i.e. a (params, apply_fun) pair
        *mask: A binary mask of shape input_shape

    Returns:
        A new layer, i.e. a (params, direct_fun, inverse_fun) triplet
    """

    def init_fun(rng, input_shape):
        scale_params, scale_apply_fun = scale
        translate_params, translate_apply_fun = translate

        def direct_fun(params, inputs, **kwargs):
            scale_params, translate_params = params

            masked_inputs = inputs * mask
            log_s = scale_apply_fun(scale_params, masked_inputs) * (1 - mask)
            t = translate_apply_fun(translate_params, masked_inputs) * (1 - mask)
            s = np.exp(log_s)

            return inputs * s + t, log_s.sum(-1, keepdims=True)

        def inverse_fun(params, inputs, **kwargs):
            scale_params, translate_params = params

            masked_inputs = inputs * mask
            log_s = scale_apply_fun(scale_params, masked_inputs) * (1 - mask)
            t = translate_apply_fun(translate_params, masked_inputs) * (1 - mask)
            s = np.exp(-log_s)

            return (inputs - t) * s, log_s.sum(-1, keepdims=True)

        return (scale_params, translate_params), direct_fun, inverse_fun

    return init_fun


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    if mask_type == "input":
        in_degrees = np.arange(in_features) % in_flow_features
    else:
        in_degrees = np.arange(in_features) % (in_flow_features - 1)

    if mask_type == "output":
        out_degrees = np.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = np.arange(out_features) % (in_flow_features - 1)

    mask = np.transpose(np.expand_dims(out_degrees, -1) >= np.expand_dims(in_degrees, 0)).astype(np.float32)
    return mask


def MaskedDense(out_dim, mask, W_init=glorot_normal(), b_init=normal()):
    init_fun, _ = Dense(out_dim, W_init, b_init)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return np.dot(inputs, W * mask) + b

    return init_fun, apply_fun


def MADE():
    """
    Args:
        *scale: A trainable scaling function, i.e. a (params, apply_fun) pair
        *translate: A trainable translation function, i.e. a (params, apply_fun) pair
        *mask: A binary mask of shape input_shape

    Returns:
        A new layer, i.e. a (params, direct_fun, inverse_fun) triplet
    """

    def init_fun(rng, input_shape):
        num_hidden = 64
        num_inputs = input_shape[-1]

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type="output")

        joiner_init_fun, joiner_apply_fun = MaskedDense(num_hidden, input_mask)
        _, joiner_params = joiner_init_fun(rng, input_shape)

        trunk_init_fun, trunk_apply_fun = stax.serial(
            Relu, MaskedDense(num_hidden, hidden_mask), Relu, MaskedDense(num_inputs * 2, output_mask)
        )
        _, trunk_params = trunk_init_fun(rng, (num_hidden,))

        def direct_fun(params, inputs, **kwargs):
            joiner_params, trunk_params = params

            h = joiner_apply_fun(joiner_params, inputs)
            m, a = trunk_apply_fun(trunk_params, h).split(2, 1)
            u = (inputs - m) * np.exp(-a)

            return u, -a.sum(-1, keepdims=True)

        def inverse_fun(params, inputs, **kwargs):
            joiner_params, trunk_params = params

            x = np.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = joiner_apply_fun(joiner_params, x)
                m, a = trunk_apply_fun(trunk_params, h).split(2, 1)
                # x[:, i_col] = inputs[:, i_col] * np.exp(a[:, i_col]) + m[:, i_col]
                x = jax.ops.index_update(
                    x, jax.ops.index[:, i_col], inputs[:, i_col] * np.exp(a[:, i_col]) + m[:, i_col]
                )

            return x, -a.sum(-1, keepdims=True)

        return (joiner_params, trunk_params), direct_fun, inverse_fun

    return init_fun


def serial(*init_funs):
    def init_fun(rng, input_shape):
        params, direct_funs, inverse_funs = [], [], []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            param, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)

            params.append(param)
            direct_funs.append(direct_fun)
            inverse_funs.append(inverse_fun)

        def direct_fun(params, inputs, **kwargs):
            rng = kwargs.pop("rng", None)
            rngs = random.split(rng, len(init_funs)) if rng is not None else (None,) * len(init_funs)
            logdets = None
            for fun, param, rng in zip(direct_funs, params, rngs):
                inputs, logdet = fun(param, inputs, rng=rng, **kwargs)
                if logdets is None:
                    logdets = logdet
                else:
                    logdets += logdet
            return inputs, logdets

        def inverse_fun(params, inputs, **kwargs):
            rng = kwargs.pop("rng", None)
            rngs = random.split(rng, len(init_funs)) if rng is not None else (None,) * len(init_funs)
            logdets = None
            for fun, param, rng in reversed(list(zip(inverse_funs, params, rngs))):
                inputs, logdet = fun(param, inputs, rng=rng, **kwargs)
                if logdets is None:
                    logdets = logdet
                else:
                    logdets += logdet
            return inputs, logdets

        return params, direct_fun, inverse_fun

    return init_fun
