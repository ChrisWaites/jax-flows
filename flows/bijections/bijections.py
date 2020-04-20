import jax.numpy as np
import numpy as onp
from jax import random, scipy
from jax.nn.initializers import orthogonal
from jax.scipy import linalg
from jax.scipy.special import expit, logit

# Each layer constructor function returns an init_fun where...
#
# init_fun: a function taking an rng key and an input shape and returns...
#     params: a pytree
#     direct_fun: a function taking params and inputs and returns...
#         outputs: the mapped inputs when the layer is applied
#         log_det_jacobian: the log of the determinant of the jacobian
#     inverse_fun: a function taking params and inputs and returns...
#         outputs: the mapped inputs when the layer is applied
#         log_det_jacobian: the log of the determinant of the jacobian


def ActNorm():
    """An implementation of an actnorm layer from `Glow: Generative Flow with Invertible 1x1 Convolutions`
    (https://arxiv.org/abs/1807.03039).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
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
            log_det_jacobian = np.full((inputs.shape[0], 1), weight.sum())

            return u, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            weight, bias = params
            u = inputs * np.exp(-weight) + bias
            log_det_jacobian = np.full((inputs.shape[0], 1), -weight.sum())

            return u, log_det_jacobian

        return (weight, bias), direct_fun, inverse_fun

    return init_fun


def AffineCoupling(scale, translate, mask):
    """An implementation of a coupling layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Args:
        scale: An ``(init_fun, apply_fun)`` pair characterizing a trainable scaling function
        translate: An ``(init_fun, apply_fun)`` pair characterizing a trainable translation function
        mask: A binary mask of shape ``input_shape``

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
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

            log_det_jacobian = log_s.sum(-1, keepdims=True)

            return inputs * s + t, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            scale_params, translate_params = params

            masked_inputs = inputs * mask
            log_s = scale_apply_fun(scale_params, masked_inputs) * (1 - mask)
            t = translate_apply_fun(translate_params, masked_inputs) * (1 - mask)
            s = np.exp(-log_s)

            log_det_jacobian = log_s.sum(-1, keepdims=True)

            return (inputs - t) * s, log_det_jacobian

        return (scale_params, translate_params), direct_fun, inverse_fun

    return init_fun


def BatchNorm(momentum=0.9):
    """An implementation of a batch normalization layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_shape, **kwargs):
        log_gamma = np.zeros(input_shape)
        beta = np.zeros(input_shape)
        eps = 1e-5

        # Which is better, keeping track of state as an edge case for batchnorm or changing
        # the entire library's interface to return a layer state and keep functions pure?
        state = {}

        def direct_fun(params, inputs, **kwargs):
            evaluation = kwargs.pop("evaluation", None)

            if "running_mean" not in state:
                state["running_mean"] = np.zeros(input_shape)
                state["running_var"] = np.ones(input_shape)

            running_mean, running_var = state["running_mean"], state["running_var"]
            log_gamma, beta = params

            if evaluation:
                mean = running_mean
                var = running_var
            else:
                batch_mean = inputs.mean(0)
                batch_var = ((inputs - batch_mean) ** 2.0).mean(0) + eps

                state["batch_mean"] = batch_mean
                state["batch_var"] = batch_var

                running_mean = (running_mean * momentum) + (batch_mean * (1 - momentum))
                running_var = (running_var * momentum) + (batch_var * (1 - momentum))

                mean = batch_mean
                var = batch_var

            x_hat = (inputs - mean) / np.sqrt(var)
            y = np.exp(log_gamma) * x_hat + beta

            log_det_jacobian = np.full((inputs.shape[0], 1), (log_gamma - 0.5 * np.log(var)).sum())

            return y, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            evaluation = kwargs.pop("evaluation", None)

            if "running_mean" not in state:
                state["running_mean"] = np.zeros(input_shape)
                state["running_var"] = np.ones(input_shape)

            running_mean, running_var = state["running_mean"], state["running_var"]
            log_gamma, beta = params

            if evaluation:
                mean = running_mean
                var = running_var
            else:
                mean = state["batch_mean"]
                var = state["batch_var"]

            x_hat = (inputs - beta) / np.exp(log_gamma)
            y = x_hat * np.sqrt(var) + mean

            log_det_jacobian = np.full((inputs.shape[0], 1), (-log_gamma + 0.5 * np.log(var)).sum())

            return y, log_det_jacobian

        return (log_gamma, beta), direct_fun, inverse_fun

    return init_fun


def Invert(bijection):
    """Inverts a tranformation so that its ``direct_fun`` is its ``inverse_fun`` and vice versa.

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_shape, **kwargs):
        params, direct_fun, inverse_fun = bijection(rng, input_shape)
        return params, inverse_fun, direct_fun

    return init_fun


def InvertibleLinear():
    """An implementation of an invertible linear layer from `Glow: Generative Flow with Invertible 1x1 Convolutions`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
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


def Logit(clip_before_logit=True):
    """Computes the logit function on a set of inputs, with sigmoid function being its inverse.

    Important note: Values passed through this layer are clipped to be within a range computable using 32 bits. This
    was done in "Cubic-Spline Flows" by Durkan et al. Technically this breaks invertibility, but it avoids
    inevitable NaNs.

    Args:
        clip_before_logit: Whether to clip values to range [1e-5, 1 - 1e-5] before being passed through logit.

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """
    return Invert(Sigmoid(clip_before_logit))


def Reverse():
    """An implementation of a reversing layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.

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


def Shuffle():
    """An implementation of a shuffling layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.

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


def Sigmoid(clip_before_logit=True):
    """Computes the sigmoid function on a set of inputs, with the logit function being its inverse.

    Important note: Values passed through this layer are clipped to be within a range computable using 32 bits. This
    was done in "Cubic-Spline Flows" by Durkan et al. Technically this breaks invertibility, but it avoids
    inevitable NaNs.

    Args:
        clip_before_logit: Whether to clip values to range [1e-5, 1 - 1e-5] before being passed through logit.

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
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


def Serial(*init_funs):
    """
    Args:
        *init_funs: Multiple bijections in sequence

    Returns:
        An ``init_fun`` mapping ``(rng, input_shape)`` to a ``(params, direct_fun, inverse_fun)`` triplet.

    Examples:
        >>> num_examples, input_shape, tol = 20, (3,), 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> inputs = random.uniform(input_rng, (num_examples,) + input_shape)
        >>> init_fun = Serial(Shuffle(), Shuffle())
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
