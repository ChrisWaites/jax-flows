import functools
import itertools
import operator as op
from tqdm import tqdm
import math

import numpy as onp
import numpy.random as npr

from jax import jit, grad, random
import jax.numpy as np

from jax.nn import (relu, log_softmax, softmax, softplus,
                    sigmoid, elu, leaky_relu, selu, gelu,
                    normalize)

from jax.nn.initializers import glorot_normal, normal, ones, zeros, orthogonal
from jax.experimental import optimizers, stax

from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax, Tanh)

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Each layer constructor function returns an (init_fun, forward_fun, backward_fun) triplet, where
#   init_fun: Takes an rng key and an input shape and returns params
#   forward_fun: Takes an input and returns (result, logdet)
#   backward_fun: Takes an input and returns (result, logdet)


def Shuffle(input_shape):
  perm = npr.permutation(onp.prod(input_shape)).reshape(input_shape)
  inv_perm = onp.argsort(perm)

  def init_fun(rng, input_shape):
    return ()

  def forward_fun(params, inputs, **kwargs):
    return inputs[:, perm], np.zeros((inputs.shape[0], 1))

  def backward_fun(params, inputs, **kwargs):
    return inputs[:, inv_perm], np.zeros((inputs.shape[0], 1))

  return init_fun, forward_fun, backward_fun


def Reverse(input_shape):
  perm = np.array(np.arange(onp.prod(input_shape))[::-1]).reshape(input_shape)
  inv_perm = np.argsort(perm)

  def init_fun(rng, input_shape):
    return ()

  def forward_fun(params, inputs, **kwargs):
    return inputs[:, perm], np.zeros((inputs.shape[0], 1))

  def backward_fun(params, inputs, **kwargs):
    return inputs[:, inv_perm], np.zeros((inputs.shape[0], 1))

  return init_fun, forward_fun, backward_fun


def CouplingLayer(scale_net, translate_net, mask):
  """
  Args:
    *scale: an (params, apply_fun) pair
    *translate: an (params, apply_fun) pair
    *mask:

  Returns:
    A new layer, meaning an (init_fun, forward_fun, backward_fun) pair
  """
  scale_params, scale_apply_fun = scale_net
  translate_params, translate_apply_fun = translate_net

  def init_fun(rng, input_shape):
    return (scale_params, translate_params)

  def forward_fun(params, inputs, **kwargs):
    scale_params, translate_params = params

    masked_inputs = inputs * mask
    log_s = scale_apply_fun(scale_params, masked_inputs) * (1 - mask)
    t = translate_apply_fun(translate_params, masked_inputs) * (1 - mask)
    s = np.exp(log_s)
    return inputs * s + t, log_s.sum(-1, keepdims=True)

  def backward_fun(params, inputs, **kwargs):
    scale_params, translate_params = params

    masked_inputs = inputs * mask
    log_s = scale_apply_fun(scale_params, masked_inputs) * (1 - mask)
    t = translate_apply_fun(translate_params, masked_inputs) * (1 - mask)
    s = np.exp(-log_s)
    return (inputs - t) * s, log_s.sum(-1, keepdims=True)

  return init_fun, forward_fun, backward_fun


"""
def BatchNormFlow(axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True, beta_init=zeros, gamma_init=ones):
  _beta_init = lambda rng, shape: beta_init(rng, shape) if center else ()
  _gamma_init = lambda rng, shape: gamma_init(rng, shape) if scale else ()
  axis = (axis,) if np.isscalar(axis) else axis

  def init_fun(rng, input_shape):
    shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
    k1, k2 = random.split(rng)
    beta, gamma = _beta_init(k1, shape), _gamma_init(k2, shape)

    return (beta, gamma)

  def forward_fun(params, x, **kwargs):
    beta, gamma = params
    ed = tuple(None if i in axis else slice(None) for i in range(np.ndim(x)))
    beta = beta[ed]
    gamma = gamma[ed]
    z = normalize(x, axis, epsilon=epsilon)

    if center and scale:
      return gamma * z + beta
    elif center:
      return z + beta
    elif scale:
      return gamma * z
    else:
      return z

  def backward_fun(params, inputs, **kwargs):
    if training:
      mean = batch_mean
      var = batch_var
    else:
      mean = running_mean
      var = running_var

    x_hat = (inputs - beta) / torch.exp(log_gamma)
    y = x_hat * var.sqrt() + mean
    return y, (-log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)

  return init_fun, forward_fun, backward_fun
"""


def serial(*layers):
  nlayers = len(layers)
  init_funs, forward_funs, backward_funs = zip(*layers)

  def init_fun(rng, input_shape):
    params = []
    for init_fun in init_funs:
      rng, layer_rng = random.split(rng)
      param = init_fun(layer_rng, input_shape)
      params.append(param)
    return params

  def forward_fun(params, inputs, **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
    logdets = None
    for fun, param, rng in zip(forward_funs, params, rngs):
      inputs, logdet = fun(param, inputs, rng=rng, **kwargs)
      if logdets is None:
          logdets = logdet
      else:
          logdets += logdet
    return inputs, logdets

  def backward_fun(params, inputs, **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
    logdets = None
    for fun, param, rng in reversed(list(zip(backward_funs, params, rngs))):
      inputs, logdet = fun(param, inputs, rng=rng, **kwargs)
      if logdets is None:
          logdets = logdet
      else:
          logdets += logdet
    return inputs, logdets

  return init_fun, forward_fun, backward_fun


def log_probs(params, forward_fun, inputs):
  u, log_jacob = forward_fun(params, inputs)
  log_probs = (-.5 * (u ** 2.) - .5 * np.log(2 * math.pi)).sum(-1, keepdims=True)
  return (log_probs + log_jacob).sum(-1, keepdims=True)


def net(rng, input_shape, hidden_dim=64, act_fun=Relu):
  init_fun, apply_fun = stax.serial(
    Dense(hidden_dim, W_init=orthogonal(), b_init=zeros), act_fun,
    Dense(hidden_dim, W_init=orthogonal(), b_init=zeros), act_fun,
    Dense(input_shape[-1], W_init=orthogonal(), b_init=zeros),
  )
  _, params = init_fun(rng, input_shape)
  return (params, apply_fun)


def mask(input_shape=(2,)):
  mask = onp.zeros(input_shape)
  mask[::2] = 1.
  return mask


n_samples = 10000
scaler = StandardScaler()
xlim, ylim = [-2, 2], [-2, 2]

X, _ = datasets.make_moons(n_samples=n_samples, noise=.05)
X = scaler.fit_transform(X)
plt.hist2d(X[:, 0], X[:, 1], bins=100, range=((-2, 2), (-2, 2)))
plt.savefig('./train.png')


rng = random.PRNGKey(0)
input_shape = (2,)

init_fun, forward_fun, backward_fun = serial(
  CouplingLayer(net(rng, input_shape, act=Tanh), net(rng, input_shape, act=Relu), mask()),
  CouplingLayer(net(rng, input_shape, act=Tanh), net(rng, input_shape, act=Relu), 1 - mask()),
  CouplingLayer(net(rng, input_shape, act=Tanh), net(rng, input_shape, act=Relu), mask()),
  CouplingLayer(net(rng, input_shape, act=Tanh), net(rng, input_shape, act=Relu), 1 - mask()),
  CouplingLayer(net(rng, input_shape, act=Tanh), net(rng, input_shape, act=Relu), mask()),
)

params = init_fun(rng, (2,))
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)


def loss(params, forward_fun, inputs):
  return -log_probs(params, forward_fun, inputs).mean()

@jit
def step(i, opt_state, batch):
  params = get_params(opt_state)
  return opt_update(i, grad(loss)(params, forward_fun, batch), opt_state)


itercount = itertools.count()
batch_size = 100
num_epochs = 1000
opt_state = opt_init(params)


for epoch in tqdm(range(num_epochs)):
  npr.shuffle(X)
  for batch_index in range(0, len(X), batch_size):
    opt_state = step(next(itercount), opt_state, X[batch_index:batch_index+batch_size])
params = get_params(opt_state)


Z = npr.normal(0, 1, (n_samples, 2))
X, _ = backward_fun(params, Z)
plt.hist2d(X[:, 0], X[:, 1], bins=100, range=((-2, 2), (-2, 2)))
plt.savefig('./synth.png')

