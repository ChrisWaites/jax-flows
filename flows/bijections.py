import jax
import jax.numpy as np
import numpy as onp
from jax import random, scipy
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu
from jax.nn.initializers import glorot_normal, normal, orthogonal
from jax.scipy import linalg
from jax.scipy.special import expit, logit

# Each layer constructor function returns an init_fun where...
#
# init_fun: a fn. taking an rng key and an input shape and returns...
#     params: a pytree
#     direct_fun: a fn. taking params and inputs and returns...
#         outputs: the mapped inputs when the layer is applied
#         log_det_jacobian: the log-determinant of the jacobian
#     inverse_fun: a fn. taking params and inputs and returns...
#         outputs: the mapped inputs when the layer is applied
#         log_det_jacobian: the log-determinant of the jacobian


def Shuffle():
    """An implementation of a shuffling layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet

    Examples:
        >>> num_examples, input_shape, tol = 20, (3,), 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> inputs = random.uniform(input_rng, (num_examples,) + input_shape)
        >>> init_fun = Shuffle()
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> onp.array_equal(inputs, reconstructed_inputs)
        True
    """

    def init_fun(rng, input_shape, **kwargs):
        perm = random.shuffle(rng, np.arange(onp.prod(input_shape))).reshape(input_shape)
        inv_perm = np.argsort(perm)

        def direct_fun(params, inputs, **kwargs):
            return inputs[:, perm], np.zeros((inputs.shape[0], 1))

        def inverse_fun(params, inputs, **kwargs):
            return inputs[:, inv_perm], np.zeros((inputs.shape[0], 1))

        return (), direct_fun, inverse_fun

    return init_fun


def Reverse():
    """An implementation of a reversing layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet

    Examples:
        >>> num_examples, input_shape, tol = 20, (3,), 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> inputs = random.uniform(input_rng, (num_examples,) + input_shape)
        >>> init_fun = Reverse()
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> onp.array_equal(inputs, reconstructed_inputs)
        True
    """

    def init_fun(rng, input_shape, **kwargs):
        perm = np.array(np.arange(onp.prod(input_shape))[::-1]).reshape(input_shape)
        inv_perm = np.argsort(perm)

        def direct_fun(params, inputs, **kwargs):
            return inputs[:, perm], np.zeros((inputs.shape[0], 1))

        def inverse_fun(params, inputs, **kwargs):
            return inputs[:, inv_perm], np.zeros((inputs.shape[0], 1))

        return (), direct_fun, inverse_fun

    return init_fun


def Invert(bijection):
    """Inverts a tranformation so that its ``direct_fun`` is its ``inverse_fun``
    and vice versa.

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet
    """

    def init_fun(rng, input_shape, **kwargs):
        params, direct_fun, inverse_fun = bijection(rng, input_shape)
        return params, inverse_fun, direct_fun

    return init_fun


def Sigmoid(clip_before_logit=True):
    """Computes the sigmoid function on a set of inputs, with the
    logit function being its inverse.

    Important note: Values passed through this layer are clipped to be within
    a range computable using 32 bits. This was done in "Cubic-Spline Flows"
    by Durkan et al. Technically this breaks invertibility, but its arguably
    better than inevitable NaNs.

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet
    """

    def init_fun(rng, input_shape, **kwargs):
        def direct_fun(params, inputs, **kwargs):
            inputs = inputs.reshape(inputs.shape[0], -1)
            log_det_jacobian = np.log(expit(inputs) * (1 - expit(inputs))).sum(-1, keepdims=True)

            return expit(inputs), log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            if clip_before_logit:
                inputs = np.clip(inputs, 1e-5, 1 - 1e-5)

            inputs = inputs.reshape(inputs.shape[0], -1)
            log_det_jacobian = -np.log(inputs - (inputs ** 2.0)).sum(-1, keepdims=True)

            return logit(inputs), log_det_jacobian

        return (), direct_fun, inverse_fun

    return init_fun


def Logit():
    """Computes the logit function on a set of inputs, with sigmoid
    function being its inverse.

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet
    """
    return Invert(Sigmoid())


def AffineCoupling(scale, translate, mask):
    """An implementation of a coupling layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Args:
        scale: An ``(init_fun, apply_fun)`` pair characterizing a trainable scaling function
        translate: An ``(init_fun, apply_fun)`` pair characterizing a trainable translation function
        mask: A binary mask of shape ``input_shape``

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet
    """

    def init_fun(rng, input_shape, **kwargs):
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


def InvertibleMM():
    """An implementation of an invertible matrix multiplication
    layer from `Glow: Generative Flow with Invertible 1x1 Convolutions`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet
    """

    def init_fun(rng, input_shape, **kwargs):
        W = orthogonal()(rng, (input_shape[-1], input_shape[-1]))

        L_mask = np.tril(np.ones(W.shape), -1)
        U_mask = L_mask.transpose()

        P, L, U = scipy.linalg.lu(W)

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(np.abs(S))

        ident = np.eye(L.shape[0])

        def direct_fun(params, inputs, **kwargs):
            L, U, log_S = params

            L = L * L_mask + ident
            U = U * U_mask + np.diag(sign_S * np.exp(log_S))
            W = P @ L @ U

            log_det_jacobian = np.full((inputs.shape[0], 1), log_S.sum())

            return inputs @ W, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            L, U, log_S = params

            L = L * L_mask + ident
            U = U * U_mask + np.diag(sign_S * np.exp(log_S))
            W = P @ L @ U

            log_det_jacobian = np.full((inputs.shape[0], 1), -log_S.sum())

            return inputs @ linalg.inv(W), log_det_jacobian

        return (L, U, log_S), direct_fun, inverse_fun

    return init_fun


def ActNorm():
    """An implementation of an activation normalization layer
    from `Glow: Generative Flow with Invertible 1x1 Convolutions`
    (https://arxiv.org/abs/1807.03039).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet
    """

    def init_fun(rng, input_shape, **kwargs):
        inputs = kwargs.pop("inputs", None)

        if not (inputs is None):
            weight = np.log(1.0 / (inputs.std(0) + 1e-12))
            bias = inputs.mean(0)
        else:
            weight = np.ones(input_shape)
            bias = np.zeros(input_shape)

        def direct_fun(params, inputs, **kwargs):
            weight, bias = params
            u = (inputs - bias) * np.exp(weight)
            log_det_jacobian = np.expand_dims(weight.sum(-1, keepdims=True), 0).repeat(inputs.shape[0], axis=0)

            return u, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            weight, bias = params
            u = inputs * np.exp(-weight) + bias
            log_det_jacobian = -np.expand_dims(weight.sum(-1, keepdims=True), 0).repeat(inputs.shape[0], axis=0)

            return u, log_det_jacobian

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
    """An implementation of `MADE: Masked Autoencoder for Distribution Estimation`
    (https://arxiv.org/abs/1502.03509).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet

    Examples:
        >>> num_examples, input_shape, tol = 20, (3,), 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> inputs = random.uniform(input_rng, (num_examples,) + input_shape)
        >>> init_fun = MADE()
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> onp.allclose(inputs, reconstructed_inputs, 1e-4)
        True
    """

    def init_fun(rng, input_shape, **kwargs):
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


def serial(*init_funs):
    """
    Args:
        inputs: An ndarray passed to each bijection's ``init_fun`` for initialization
        *init_funs: Multiple bijections in sequence

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a
        ``(params, direct_fun, inverse_fun)`` triplet

    Examples:
        >>> num_examples, input_shape, tol = 20, (3,), 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> inputs = random.uniform(input_rng, (num_examples,) + input_shape)
        >>> init_fun = serial(Shuffle(), Shuffle())
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_shape)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> onp.array_equal(inputs, reconstructed_inputs)
        True
    """

    def init_fun(rng, input_shape, **kwargs):
        inputs = kwargs.pop("inputs", None)

        all_params, direct_funs, inverse_funs = [], [], []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            param, direct_fun, inverse_fun = init_fun(layer_rng, input_shape, inputs=inputs)

            if not (inputs is None):
                inputs = direct_fun(param, inputs)

            all_params.append(param)
            direct_funs.append(direct_fun)
            inverse_funs.append(inverse_fun)

        def feed_forward(params, apply_funs, inputs):
            log_det_jacobians = None
            for apply_fun, param in zip(apply_funs, params):
                inputs, log_det_jacobian = apply_fun(param, inputs, **kwargs)
                if log_det_jacobians is None:
                    log_det_jacobians = log_det_jacobian
                else:
                    log_det_jacobians += log_det_jacobian
            return inputs, log_det_jacobians

        def direct_fun(params, inputs, **kwargs):
            return feed_forward(params, direct_funs, inputs)

        def inverse_fun(params, inputs, **kwargs):
            return feed_forward(reversed(params), reversed(inverse_funs), inputs)

        return all_params, direct_fun, inverse_fun

    return init_fun
