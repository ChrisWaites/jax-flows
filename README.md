<img align="right" width="300" src="assets/flows.gif">

# Normalizing Flows in JAX

Implementations of normalizing flows (RealNVP, GLOW, MAF) in the <a href="https://github.com/google/jax/">JAX</a> deep learning framework.</p>

<a href="https://circleci.com/gh/ChrisWaites/jax-flows">
    <img alt="Build" src="https://img.shields.io/circleci/build/github/ChrisWaites/jax-flows/master">
</a>
<a href="https://github.com/ChrisWaites/jax-flows/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/ChrisWaites/jax-flows.svg?color=blue">
</a>
<a href="https://ChrisWaites.co/jax-flows/index.html">
    <img alt="Documentation" src="https://img.shields.io/website/http/ChrisWaites.co/jax-flows/index.html.svg?down_color=red&down_message=offline&up_message=online">
</a>
<a href="https://github.com/ChrisWaites/jax-flows/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/release/ChrisWaites/jax-flows.svg">
</a>

## What are normalizing flows?

Normalizing flow models are _generative models_. That is, they infer the probability distribution of a given dataset. With that distribution we can do a number of interesting things, namely query the likelihood of given points as well as sample new realistic points.

<!---
How is are these things achieved? Well, we learn a function <img src="https://render.githubusercontent.com/render/math?math=f_{\theta}"> characterized by a parameter vector <img src="https://render.githubusercontent.com/render/math?math=\theta"> with an inverse <img src="https://render.githubusercontent.com/render/math?math=f^{-1}_{\theta}">. If X is our approximated distribution and Z is some known distribution we choose (say, the multivariate normal distribution), we're simply going to define X as f_\theta(Z).
-->

## How are things structured?

### Transformations

A `transformation` is a parameterized invertible function.

```python
init_fun = flows.MADE()

params, direct_fun, inverse_fun = init_fun(rng, input_shape)

# Transform some inputs
transformed_inputs, log_det_direct = direct_fun(params, inputs)

# Reconstruct original inputs
reconstructed_inputs, log_det_inverse = inverse_fun(params, inputs)

assert np.array_equal(inputs, reconstructed_inputs)
```

We can construct a larger meta-transformation by composing a sequence of sub-transformations using `flows.serial`. The resulting transformation adheres to the exact same interface and is indistinguishable from any other regular transformation.

```python
init_fun = flows.serial(
  flows.MADE(),
  flows.BatchNorm(),
  flows.Reverse()
)

params, direct_fun, inverse_fun = init_fun(rng, input_shape)
```

### Distributions

A `distribution` has a similarly simple interface. It is characterized by a set of parameters, a function for querying the log of the pdf at a given point, and a sampling function.

```python
init_fun = Normal()

params, log_pdf, sample = init_fun(rng, input_shape)

log_pdfs = log_pdf(params, inputs)

samples = sample(rng, params, num_samples)
```

### Normalizing Flow Models

Under this definition, a normalizing flow model is just a `distribution`. But to retrieve one, we have to give it a transformation and another prior distribution.

```python
transformation = flows.serial(
  flows.MADE(),
  flows.BatchNorm(),
  flows.Reverse(),
  flows.MADE(),
  flows.BatchNorm(),
  flows.Reverse(),
)

prior = Normal()

init_fun = flows.Flow(transformation, prior)

params, log_pdf, sample = init_fun(rng, input_shape)
```

### How do I train a model?

To train our model, we would typically define an appropriate loss function and parameter update step.

```python
def loss(params, inputs):
  return -log_pdf(params, inputs).mean()

@jit
def step(i, opt_state, inputs):
  params = get_params(opt_state)
  return opt_update(i, grad(loss)(params, inputs), opt_state)
```

Given these, we can go forward and execute a standard JAX training loop.

```python
batch_size = 32

itercount = itertools.count()
for epoch in range(num_epochs):
  npr.shuffle(X)
  for batch_index in range(0, len(X), batch_size):
    opt_state = step(next(itercount), opt_state, X[batch_index:batch_index+batch_size])

optimized_params = get_params(opt_state)
```

Now that we have our trained model parameters, we can query and sample as regular.

```python
log_pdfs = log_pdf(optimized_params, inputs)

samples = sample(rng, optimized_params, num_samples)
```

_Magic!_

## Interested in contributing?

Yay! Check out our contributing guidelines in `.github/CONTRIBUTING.md`.

## Inspiration

This repository is largely modeled after the [`pytorch-flows`](https://github.com/ikostrikov/pytorch-flows) repository by [Ilya Kostrikov
](https://github.com/ikostrikov).

The implementations are modeled after the work of the following papers.

  > [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)\
  > Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio\
  > _arXiv:1605.08803_

  > [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)\
  > Diederik P. Kingma, Prafulla Dhariwal\
  > _arXiv:1807.03039_

  > [Flow++: Improving Flow-Based Generative Models
  with Variational Dequantization and Architecture Design](https://openreview.net/forum?id=Hyg74h05tX)\
  > Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, Pieter Abbeel\
  > _OpenReview:Hyg74h05tX_

  > [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057)\
  > George Papamakarios, Theo Pavlakou, Iain Murray\
  > _arXiv:1705.07057_

