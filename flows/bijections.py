import jax
import jax.numpy as np
import numpy as onp
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu
from jax.nn.initializers import glorot_normal, normal
from jax.scipy.special import expit, logit


def Shuffle():
    """An implementation of a shuffling layer from RealNVP
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet

    Examples:
        >>> num_examples, input_shape, tol = 100, (3,), 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> init_fun = Shuffle()
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)
        >>> inputs = random.uniform(input_rng, (num_examples,) + input_shape)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> onp.array_equal(inputs, reconstructed_inputs)
        True
    """

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
    """An implementation of a reversing layer from RealNVP
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet

    Examples:
        >>> num_examples, input_shape, tol = 100, (3,), 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> init_fun = Reverse()
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)
        >>> inputs = random.uniform(input_rng, (num_examples,) + input_shape)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> onp.array_equal(inputs, reconstructed_inputs)
        True
    """

    def init_fun(rng, input_shape):
        perm = np.array(np.arange(onp.prod(input_shape))[::-1]).reshape(input_shape)
        inv_perm = np.argsort(perm)

        def direct_fun(params, inputs, **kwargs):
            return inputs[:, perm], np.zeros((inputs.shape[0], 1))

        def inverse_fun(params, inputs, **kwargs):
            return inputs[:, inv_perm], np.zeros((inputs.shape[0], 1))

        return (), direct_fun, inverse_fun

    return init_fun


def Invert(bijection):
    """Inverts a tranformation so that its `direct_fun` is its `inverse_fun`
    and vice versa.

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet
    """

    def init_fun(rng, input_shape):
        params, direct_fun, inverse_fun = bijection(rng, input_shape)
        return params, inverse_fun, direct_fun

    return init_fun


def Sigmoid(clip_before_logit=True):
    """Computes the sigmoid (expit) function on a set of inputs, with the
    logit function being its inverse.

    Important note: Values passed through this layer are clipped to be within
    a range computable using 32 bits. This was done in "Cubic-Spline Flows"
    by Durkan et al. Technically this breaks invertibility, but its arguably
    better than inevitable NaNs.

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet
    """

    def init_fun(rng, input_shape):
        def direct_fun(params, inputs, **kwargs):
            inputs = inputs.reshape(inputs.shape[0], -1)
            log_det = np.log(expit(inputs) * (1 - expit(inputs))).sum(-1, keepdims=True)

            return expit(inputs), log_det

        def inverse_fun(params, inputs, **kwargs):
            if clip_before_logit:
                inputs = np.clip(inputs, 1e-5, 1 - 1e-5)

            inputs = inputs.reshape(inputs.shape[0], -1)
            log_det = -np.log(inputs - (inputs ** 2.0)).sum(-1, keepdims=True)

            return logit(inputs), log_det

        return (), direct_fun, inverse_fun

    return init_fun


def Logit():
    """Computes the logit function on a set of inputs, with the sigmoid (expit)
    function being its inverse.

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet
    """
    return Invert(Sigmoid())


def AffineCoupling(scale, translate, mask):
    """An implementation of a coupling layer from RealNVP
    (https://arxiv.org/abs/1605.08803).

    Args:
        scale: An `(init_fun, apply_fun)` pair characterizing a trainable scaling function
        translate: An `(init_fun, apply_fun)` pair characterizing a trainable translation function
        mask: A binary mask of shape `input_shape`

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet
    """

    def init_fun(rng, input_shape):
        scale_rng, translate_rng = random.split(rng)

        scale_init_fun, scale_apply_fun = scale
        _, scale_params = scale_init_fun(scale_rng, input_shape)

        translate_init_fun, translate_apply_fun = translate
        _, translate_params = translate_init_fun(translate_rng, input_shape)

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


def ActNorm():
    """An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet
    """

    def init_fun(rng, input_shape):
        weight = np.ones(input_shape)
        bias = np.zeros(input_shape)

        def direct_fun(params, inputs, **kwargs):
            weight, bias = params
            u = (inputs - bias) * np.exp(weight)
            log_det = np.expand_dims(weight.reshape(weight.shape[0], -1).sum(-1), 0).repeat(inputs.shape[0], axis=0)
            return u, log_det

        def inverse_fun(params, inputs, **kwargs):
            weight, bias = params
            u = inputs * np.exp(-weight) + bias
            log_det = np.expand_dims(weight.reshape(weight.shape[0], -1).sum(-1), 0).repeat(inputs.shape[0], axis=0)
            return u, log_det

        return (weight, bias), direct_fun, inverse_fun

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

    mask = np.expand_dims(out_degrees, -1) >= np.expand_dims(in_degrees, 0)
    return np.transpose(mask).astype(np.float32)


def MaskedDense(out_dim, mask, W_init=glorot_normal(), b_init=normal()):
    init_fun, _ = Dense(out_dim, W_init, b_init)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return np.dot(inputs, W * mask) + b

    return init_fun, apply_fun


def MADE():
    """An implementation of MADE (https://arxiv.org/abs/1502.03509).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet

    Examples:
        >>> num_examples, input_shape, tol = 100, (3,), 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> init_fun = MADE()
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)
        >>> inputs = random.uniform(input_rng, (num_examples,) + input_shape)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> onp.allclose(inputs, reconstructed_inputs, 1e-4)
        True
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
                x = jax.ops.index_update(
                    x, jax.ops.index[:, i_col], inputs[:, i_col] * np.exp(a[:, i_col]) + m[:, i_col]
                )

            return x, -a.sum(-1, keepdims=True)

        return (joiner_params, trunk_params), direct_fun, inverse_fun

    return init_fun


def serial(*init_funs):
    """
    Args:
        *init_funs: Multiple bijections in sequence

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet

    Examples:
        >>> num_examples, input_shape, tol = 100, (3,), 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> init_fun = serial(Shuffle(), Shuffle(), Shuffle())
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)
        >>> inputs = random.uniform(input_rng, (num_examples,) + input_shape)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> onp.array_equal(inputs, reconstructed_inputs)
        True
    """

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
            log_dets = None
            for fun, param, rng in zip(direct_funs, params, rngs):
                inputs, log_det = fun(param, inputs, rng=rng, **kwargs)
                if log_dets is None:
                    log_dets = log_det
                else:
                    log_dets += log_det
            return inputs, log_dets

        def inverse_fun(params, inputs, **kwargs):
            rng = kwargs.pop("rng", None)
            rngs = random.split(rng, len(init_funs)) if rng is not None else (None,) * len(init_funs)
            log_dets = None
            for fun, param, rng in reversed(list(zip(inverse_funs, params, rngs))):
                inputs, log_det = fun(param, inputs, rng=rng, **kwargs)
                if log_dets is None:
                    log_dets = log_det
                else:
                    log_dets += log_det
            return inputs, log_dets

        return params, direct_fun, inverse_fun

    return init_fun
