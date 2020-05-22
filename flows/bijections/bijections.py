import jax.numpy as np
import jax.scipy.special as spys
from jax import random, scipy
from jax.nn.initializers import orthogonal
from jax.scipy import linalg

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
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        init_inputs = kwargs.pop("init_inputs", None)

        if not (init_inputs is None):
            log_weight = np.log(1.0 / (init_inputs.std(0) + 1e-6))
            bias = init_inputs.mean(0)
        else:
            log_weight = np.zeros(input_dim)
            bias = np.zeros(input_dim)

        def direct_fun(params, inputs, **kwargs):
            log_weight, bias = params
            outputs = (inputs - bias) * np.exp(log_weight)
            log_det_jacobian = np.full(inputs.shape[:1], log_weight.sum())
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            log_weight, bias = params
            outputs = inputs * np.exp(-log_weight) + bias
            log_det_jacobian = np.full(inputs.shape[:1], -log_weight.sum())
            return outputs, log_det_jacobian

        return (log_weight, bias), direct_fun, inverse_fun

    return init_fun


def AffineCouplingSplit(scale, translate):
    """An implementation of a coupling layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Args:
        scale: An ``(params, apply_fun)`` pair characterizing a trainable scaling function
        translate: An ``(params, apply_fun)`` pair characterizing a trainable translation function

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        cutoff = input_dim // 2

        scale_rng, rng = random.split(rng)
        scale_params, scale_apply_fun = scale(scale_rng, cutoff, input_dim - cutoff)

        translate_rng, rng = random.split(rng)
        translate_params, translate_apply_fun = translate(translate_rng, cutoff, input_dim - cutoff)

        def direct_fun(params, inputs, **kwargs):
            scale_params, translate_params = params
            lower, upper = inputs[:, :cutoff], inputs[:, cutoff:]

            log_weight = scale_apply_fun(scale_params, lower)
            bias = translate_apply_fun(translate_params, lower)
            upper = upper * np.exp(log_weight) + bias

            outputs = np.concatenate([lower, upper], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            scale_params, translate_params = params
            lower, upper = inputs[:, :cutoff], inputs[:, cutoff:]

            log_weight = scale_apply_fun(scale_params, lower)
            bias = translate_apply_fun(translate_params, lower)
            upper = (upper - bias) * np.exp(-log_weight)

            outputs = np.concatenate([lower, upper], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs, log_det_jacobian

        return (scale_params, translate_params), direct_fun, inverse_fun

    return init_fun


def AffineCoupling(transform):
    """An implementation of a coupling layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Args:
        net: An ``(params, apply_fun)`` pair characterizing a trainable translation function

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        cutoff = input_dim // 2
        params, apply_fun = transform(rng, cutoff, 2 * (input_dim - cutoff))

        def direct_fun(params, inputs, **kwargs):
            lower, upper = inputs[:, :cutoff], inputs[:, cutoff:]

            log_weight, bias = apply_fun(params, lower).split(2, axis=1)
            upper = upper * np.exp(log_weight) + bias

            outputs = np.concatenate([lower, upper], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            lower, upper = inputs[:, :cutoff], inputs[:, cutoff:]

            log_weight, bias = apply_fun(params, lower).split(2, axis=1)
            upper = (upper - bias) * np.exp(-log_weight)

            outputs = np.concatenate([lower, upper], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs, log_det_jacobian

        return params, direct_fun, inverse_fun

    return init_fun


def BatchNorm(momentum=0.9):
    """An implementation of a batch normalization layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        log_weight = np.zeros(input_dim)
        bias = np.zeros(input_dim)
        eps = 1e-5

        # Which is better, keeping track of state as an edge case for batchnorm or changing
        # the entire library's interface to return a layer state and keep functions pure?
        state = {}

        def direct_fun(params, inputs, **kwargs):
            evaluation = kwargs.pop("evaluation", None)
            log_weight, bias = params

            if "running_mean" not in state:
                state["running_mean"] = np.zeros(input_dim)
                state["running_var"] = np.ones(input_dim)
            running_mean, running_var = state["running_mean"], state["running_var"]

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

            outputs = x_hat * np.exp(log_weight) + bias
            log_det_jacobian = np.full((inputs.shape[0],), (log_weight - 0.5 * np.log(var)).sum())
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            evaluation = kwargs.pop("evaluation", None)
            log_weight, bias = params

            if "running_mean" not in state:
                state["running_mean"] = np.zeros(input_dim)
                state["running_var"] = np.ones(input_dim)
            running_mean, running_var = state["running_mean"], state["running_var"]

            if evaluation:
                mean = running_mean
                var = running_var
            else:
                mean = state["batch_mean"]
                var = state["batch_var"]

            x_hat = (inputs - bias) * np.exp(-log_weight)

            outputs = x_hat * np.sqrt(var) + mean
            log_det_jacobian = np.full((inputs.shape[0],), (-log_weight + 0.5 * np.log(var)).sum())
            return outputs, log_det_jacobian

        return (log_weight, bias), direct_fun, inverse_fun

    return init_fun


def Invert(bijection):
    """Inverts a tranformation so that its ``direct_fun`` is its ``inverse_fun`` and vice versa.

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        params, direct_fun, inverse_fun = bijection(rng, input_dim)
        return params, inverse_fun, direct_fun

    return init_fun


def FixedInvertibleLinear():
    """An implementation of an invertible linear layer from `Glow: Generative Flow with Invertible 1x1 Convolutions`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        W = orthogonal()(rng, (input_dim, input_dim))
        W_inv = linalg.inv(W)
        W_log_det = np.linalg.slogdet(W)[-1]

        def direct_fun(params, inputs, **kwargs):
            outputs = inputs @ W
            log_det_jacobian = np.full(inputs.shape[:1], W_log_det)
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs @ W_inv
            log_det_jacobian = np.full(inputs.shape[:1], -W_log_det)
            return outputs, log_det_jacobian

        return (), direct_fun, inverse_fun

    return init_fun


def InvertibleLinear():
    """An implementation of an invertible linear layer from `Glow: Generative Flow with Invertible 1x1 Convolutions`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        W = orthogonal()(rng, (input_dim, input_dim))
        P, L, U = scipy.linalg.lu(W)
        S = np.diag(U)
        U = np.triu(U, 1)
        identity = np.eye(input_dim)

        def direct_fun(params, inputs, **kwargs):
            L, U, S = params
            L = np.tril(L, -1) + identity
            U = np.triu(U, 1)
            W = P @ L @ (U + np.diag(S))

            outputs = inputs @ W
            log_det_jacobian = np.full(inputs.shape[:1], np.log(np.abs(S)).sum())
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            L, U, S = params
            L = np.tril(L, -1) + identity
            U = np.triu(U, 1)
            W = P @ L @ (U + np.diag(S))

            outputs = inputs @ linalg.inv(W)
            log_det_jacobian = np.full(inputs.shape[:1], -np.log(np.abs(S)).sum())
            return outputs, log_det_jacobian

        return (L, U, S), direct_fun, inverse_fun

    return init_fun


def Logit(clip_before_logit=True):
    """Computes the logit function on a set of inputs, with sigmoid function being its inverse.

    Important note: Values passed through this layer are clipped to be within a range computable using 32 bits. This
    was done in "Cubic-Spline Flows" by Durkan et al. Technically this breaks invertibility, but it avoids
    inevitable NaNs.

    Args:
        clip_before_logit: Whether to clip values to range [1e-5, 1 - 1e-5] before being passed through logit.

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """
    return Invert(Sigmoid(clip_before_logit))


def Reverse():
    """An implementation of a reversing layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.

    Examples:
        >>> num_examples, input_dim, tol = 20, 3, 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> inputs = random.uniform(input_rng, (num_examples, input_dim))
        >>> init_fun = Reverse()
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_dim)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> np.allclose(inputs, reconstructed_inputs).item()
        True
    """

    def init_fun(rng, input_dim, **kwargs):
        perm = np.arange(input_dim)[::-1]

        def direct_fun(params, inputs, **kwargs):
            return inputs[:, perm], np.zeros(inputs.shape[:1])

        def inverse_fun(params, inputs, **kwargs):
            return inputs[:, perm], np.zeros(inputs.shape[:1])

        return (), direct_fun, inverse_fun

    return init_fun


def Shuffle():
    """An implementation of a shuffling layer from `Density Estimation Using RealNVP`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.

    Examples:
        >>> num_examples, input_dim, tol = 20, 3, 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> inputs = random.uniform(input_rng, (num_examples, input_dim))
        >>> init_fun = Shuffle()
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_dim)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> np.allclose(inputs, reconstructed_inputs).item()
        True
    """

    def init_fun(rng, input_dim, **kwargs):
        perm = random.permutation(rng, np.arange(input_dim))
        inv_perm = np.argsort(perm)

        def direct_fun(params, inputs, **kwargs):
            return inputs[:, perm], np.zeros(inputs.shape[:1])

        def inverse_fun(params, inputs, **kwargs):
            return inputs[:, inv_perm], np.zeros(inputs.shape[:1])

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
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim, **kwargs):
        def direct_fun(params, inputs, **kwargs):
            outputs = spys.expit(inputs)
            log_det_jacobian = np.log(spys.expit(inputs) * (1 - spys.expit(inputs))).sum(-1)
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):
            if clip_before_logit:
                inputs = np.clip(inputs, 1e-5, 1 - 1e-5)

            outputs = spys.logit(inputs)
            log_det_jacobian = -np.log(inputs - np.square(inputs)).sum(-1)
            return outputs, log_det_jacobian

        return (), direct_fun, inverse_fun

    return init_fun


def Serial(*init_funs):
    """
    Args:
        *init_funs: Multiple bijections in sequence

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.

    Examples:
        >>> num_examples, input_dim, tol = 20, 3, 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> inputs = random.uniform(input_rng, (num_examples, input_dim))
        >>> init_fun = Serial(Shuffle(), Shuffle())
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_dim)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> np.allclose(inputs, reconstructed_inputs).item()
        True
    """

    def init_fun(rng, input_dim, **kwargs):
        init_inputs = kwargs.pop("init_inputs", None)

        all_params, direct_funs, inverse_funs = [], [], []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            param, direct_fun, inverse_fun = init_fun(layer_rng, input_dim, init_inputs=init_inputs)

            all_params.append(param)
            direct_funs.append(direct_fun)
            inverse_funs.append(inverse_fun)

            if not (init_inputs is None):
                init_inputs = direct_fun(param, init_inputs)[0]

        def feed_forward(params, apply_funs, inputs):
            log_det_jacobians = np.zeros(inputs.shape[:1])
            for apply_fun, param in zip(apply_funs, params):
                inputs, log_det_jacobian = apply_fun(param, inputs, **kwargs)
                log_det_jacobians += log_det_jacobian
            return inputs, log_det_jacobians

        def direct_fun(params, inputs, **kwargs):
            return feed_forward(params, direct_funs, inputs)

        def inverse_fun(params, inputs, **kwargs):
            return feed_forward(reversed(params), reversed(inverse_funs), inputs)

        return all_params, direct_fun, inverse_fun

    return init_fun
