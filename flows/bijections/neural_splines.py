import jax.numpy as np
import numpy as onp
from jax import nn, ops, random
from jax.experimental import stax

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations = ops.index_add(bin_locations, ops.index[..., -1], eps)  # bin_locations[..., -1] += eps
    return np.sum(inputs[..., None] >= bin_locations, axis=-1) - 1


def unconstrained_RQS(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = np.zeros_like(inputs)
    logabsdet = np.zeros_like(inputs)

    unnormalized_derivatives = np.pad(
        unnormalized_derivatives, [(0, 0)] * (len(unnormalized_derivatives.shape) - 1) + [(1, 1)]
    )
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives = ops.index_update(
        unnormalized_derivatives, ops.index[..., 0], constant
    )  # unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives = ops.index_update(
        unnormalized_derivatives, ops.index[..., -1], constant
    )  # unnormalized_derivatives[..., -1] = constant

    outputs = ops.index_update(
        outputs, ops.index[outside_interval_mask], inputs[outside_interval_mask]
    )  # outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet = ops.index_update(
        logabsdet, ops.index[outside_interval_mask], 0
    )  # logabsdet[outside_interval_mask] = 0

    outs, logdets = RQS(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    outputs = ops.index_update(outputs, ops.index[inside_interval_mask], outs)  # outputs[inside_interval_mask] = outs
    logabsdet = ops.index_update(
        logabsdet, ops.index[inside_interval_mask], logdets
    )  # logabsdet[inside_interval_mask] = logdets

    return outputs, logabsdet


def RQS(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if np.min(inputs) < left or np.max(inputs) > right:
        raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = np.cumsum(widths, axis=-1)
    cumwidths = np.pad(
        cumwidths, [(0, 0)] * (len(cumwidths.shape) - 1) + [(1, 0)], mode="constant", constant_values=0.0
    )
    cumwidths = (right - left) * cumwidths + left
    cumwidths = ops.index_update(cumwidths, ops.index[..., 0], left)  # cumwidths[..., 0] = left
    cumwidths = ops.index_update(cumwidths, ops.index[..., -1], right)  # cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + nn.softplus(unnormalized_derivatives)

    heights = nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = np.cumsum(heights, axis=-1)
    cumheights = np.pad(
        cumheights, [(0, 0)] * (len(cumheights.shape) - 1) + [(1, 0)], mode="constant", constant_values=0.0
    )
    cumheights = (top - bottom) * cumheights + bottom
    cumheights = ops.index_update(cumheights, ops.index[..., 0], bottom)  # cumheights[..., 0] = bottom
    cumheights = ops.index_update(cumheights, ops.index[..., -1], top)  # cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = np.take_along_axis(cumwidths, bin_idx, -1)[..., 0]  # cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = np.take_along_axis(widths, bin_idx, -1)[..., 0]  # widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = np.take_along_axis(cumheights, bin_idx, -1)[..., 0]  # cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = np.take_along_axis(delta, bin_idx, -1)[..., 0]  # delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = np.take_along_axis(derivatives, bin_idx, -1)[..., 0]  # derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = np.take_along_axis(
        derivatives[..., 1:], bin_idx, -1
    )  # derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = np.take_along_axis(heights, bin_idx, -1)[..., 0]  # heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = np.square(b) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - np.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = np.square(input_delta) * (
            input_derivatives_plus_one * np.square(root)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * np.square(1 - root)
        )
        logabsdet = np.log(derivative_numerator) - 2 * np.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * np.square(theta) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = np.square(input_delta) * (
            input_derivatives_plus_one * np.square(theta)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * np.square(1 - theta)
        )
        logabsdet = np.log(derivative_numerator) - 2 * np.log(denominator)
        return outputs, logabsdet


def FCNN(out_dim, hidden_dim):
    return stax.serial(stax.Dense(hidden_dim), stax.Tanh, stax.Dense(hidden_dim), stax.Tanh, stax.Dense(out_dim),)


"""
def NeuralSplineAutoregressive(dim, K=5, B=3, hidden_dim=8, base_network=FCNN):

    def init_fun(rng, input_shape, **kwargs):
        layers = nn.ModuleList()
        init_param = nn.Parameter(np.Tensor(3 * K - 1))
        for i in range(1, dim):
            layers += [base_network(i, 3 * K - 1, hidden_dim)]
        reset_parameters()

        def reset_parameters():
            init.uniform_(init_param, -1 / 2, 1 / 2)

        def direct_fun(params, x):
            z = np.zeros_like(x)
            log_det = np.zeros(z.shape[0])
            for i in range(dim):
                if i == 0:
                    init_param = init_param.expand(x.shape[0], 3 * K - 1)
                    W, H, D = np.split(init_param, K, dim=1)
                else:
                    out = layers[i - 1](x[:, :i])
                    W, H, D = np.split(out, K, dim=1)
                W, H = nn.softmax(W, axis=1), nn.softmax(H, axis=1)
                W, H = 2 * B * W, 2 * B * H
                D = nn.softplus(D)
                z[:, i], ld = unconstrained_RQS(x[:, i], W, H, D, inverse=False, tail_bound=B)
                log_det += ld
            return z, log_det

        def inverse_fun(params, z):
            x = np.zeros_like(z)
            log_det = np.zeros(x.shape[0])
            for i in range(dim):
                if i == 0:
                    init_param = init_param.expand(x.shape[0], 3 * K - 1)
                    W, H, D = np.split(init_param, K, dim=1)
                else:
                    out = layers[i - 1](x[:, :i])
                    W, H, D = np.split(out, K, dim=1)
                W, H = nn.softmax(W, axis=1), nn.softmax(H, axis=1)
                W, H = 2 * B * W, 2 * B * H
                D = nn.softplus(D)
                x[:, i], ld = unconstrained_RQS(z[:, i], W, H, D, inverse=True, tail_bound=B)
                log_det += ld
            return x, log_det

        return (), direct_fun, inverse_fun

    return init_fun
"""


def NeuralSplineCoupling(K=5, B=3, hidden_dim=8, network=FCNN):
    def init_fun(rng, dim, **kwargs):
        f1_rng, f2_rng = random.split(rng)

        f1_init_fun, f1_apply_fun = network((3 * K - 1) * dim // 2, hidden_dim)
        _, f1_params = f1_init_fun(f1_rng, (dim // 2,))

        f2_init_fun, f2_apply_fun = network((3 * K - 1) * dim // 2, hidden_dim)
        _, f2_params = f2_init_fun(f2_rng, (dim // 2,))

        def direct_fun(params, x):
            log_det = np.zeros(x.shape[0])
            idx = dim // 2
            lower, upper = x[:, :idx], x[:, idx:]
            out = f1_apply_fun(f1_params, lower).reshape(-1, dim // 2, 3 * K - 1)
            W, H, D = onp.array_split(out, 3, axis=2)
            W, H = nn.softmax(W, axis=2), nn.softmax(H, axis=2)
            W, H = 2 * B * W, 2 * B * H
            D = nn.softplus(D)
            upper, ld = unconstrained_RQS(upper, W, H, D, inverse=False, tail_bound=B)
            log_det += np.sum(ld, axis=1)
            out = f2_apply_fun(f2_params, upper).reshape(-1, dim // 2, 3 * K - 1)
            W, H, D = onp.array_split(out, 3, axis=2)
            W, H = nn.softmax(W, axis=2), nn.softmax(H, axis=2)
            W, H = 2 * B * W, 2 * B * H
            D = nn.softplus(D)
            lower, ld = unconstrained_RQS(lower, W, H, D, inverse=False, tail_bound=B)
            log_det += np.sum(ld, axis=1)
            return np.concatenate([lower, upper], axis=1), log_det.reshape((x.shape[0],))

        def inverse_fun(params, z):
            log_det = np.zeros(z.shape[0])
            idx = dim // 2
            lower, upper = z[:, :idx], z[:, idx:]
            out = f2_apply_fun(f2_params, upper).reshape(-1, dim // 2, 3 * K - 1)
            W, H, D = onp.array_split(out, 3, axis=2)
            W, H = nn.softmax(W, axis=2), nn.softmax(H, axis=2)
            W, H = 2 * B * W, 2 * B * H
            D = nn.softplus(D)
            lower, ld = unconstrained_RQS(lower, W, H, D, inverse=True, tail_bound=B)
            log_det += np.sum(ld, axis=1)
            out = f1_apply_fun(f1_params, lower).reshape(-1, dim // 2, 3 * K - 1)
            W, H, D = onp.array_split(out, 3, axis=2)
            W, H = nn.softmax(W, axis=2), nn.softmax(H, axis=2)
            W, H = 2 * B * W, 2 * B * H
            D = nn.softplus(D)
            upper, ld = unconstrained_RQS(upper, W, H, D, inverse=True, tail_bound=B)
            log_det += np.sum(ld, axis=1)
            return np.concatenate([lower, upper], axis=1), log_det.reshape((z.shape[0],))

        return (f1_params, f2_params), direct_fun, inverse_fun

    return init_fun
