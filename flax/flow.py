from .distributions import Normal


def Flow(rng, input_shape, transformation, prior=Normal()):
    params, direct_fun, inverse_fun = transformation(rng, input_shape)
    prior_logpdf, prior_sample = prior

    def logpdf(params, inputs):
        u, log_jacob = direct_fun(params, inputs)
        log_probs = prior_logpdf(u)
        return (log_probs + log_jacob).sum(-1, keepdims=True)

    def sample(rng, params, num_samples=1):
        return inverse_fun(params, prior_sample(rng, input_shape, num_samples))[0]

    return params, logpdf, sample
