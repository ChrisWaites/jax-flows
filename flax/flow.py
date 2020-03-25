from .distributions import Gaussian


def Flow(transformation_init_fun, rng, input_shape, prior=Gaussian()):
    params, normalizing_fun, generative_fun = transformation_init_fun(rng, input_shape)
    pdf, prior_sample = prior

    def log_prob(params, inputs):
        u, log_jacob = normalizing_fun(params, inputs)
        log_probs = pdf(u)
        return (log_probs + log_jacob).sum(-1, keepdims=True)

    def sample(params, num_samples):
        return generative_fun(params, prior_sample(input_shape, num_samples))[0]

    return params, log_prob, sample
