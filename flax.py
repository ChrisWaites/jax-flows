import functools
import itertools
import operator as op
from tqdm import tqdm
import math

import numpy as onp
import numpy.random as npr

import jax
from jax import jit, grad, random
import jax.numpy as np

from jax.nn.initializers import glorot_normal, normal, ones, zeros, orthogonal
from jax.experimental import optimizers, stax
from jax.experimental.stax import Relu, Tanh


def Shuffle():
  def init_fun(rng, input_shape):
    perm = npr.permutation(onp.prod(input_shape)).reshape(input_shape)
    inv_perm = onp.argsort(perm)

    def normalizing_fun(params, inputs, **kwargs):
      return inputs[:, perm], np.zeros((inputs.shape[0], 1))

    def generative_fun(params, inputs, **kwargs):
      return inputs[:, inv_perm], np.zeros((inputs.shape[0], 1))

    return (), normalizing_fun, generative_fun
  return init_fun


def Reverse():
  def init_fun(rng, input_shape):
    perm = np.array(np.arange(onp.prod(input_shape))[::-1]).reshape(input_shape)
    inv_perm = np.argsort(perm)

    def normalizing_fun(params, inputs, **kwargs):
      return inputs[:, perm], np.zeros((inputs.shape[0], 1))

    def generative_fun(params, inputs, **kwargs):
      return inputs[:, inv_perm], np.zeros((inputs.shape[0], 1))

    return (), normalizing_fun, generative_fun
  return init_fun


def CouplingLayer(scale, translate, mask):
  """
  Args:
    *scale: A trainable scaling function, i.e. a (params, apply_fun) pair
    *translate: A trainable translation function, i.e. a (params, apply_fun) pair
    *mask: A binary mask of shape input_shape

  Returns:
    A new layer, i.e. a (params, normalizing_fun, generative_fun) triplet
  """
  def init_fun(rng, input_shape):
    scale_params, scale_apply_fun = scale
    translate_params, translate_apply_fun = translate

    def normalizing_fun(params, inputs, **kwargs):
      scale_params, translate_params = params

      masked_inputs = inputs * mask
      log_s = scale_apply_fun(scale_params, masked_inputs) * (1 - mask)
      t = translate_apply_fun(translate_params, masked_inputs) * (1 - mask)
      s = np.exp(log_s)

      return inputs * s + t, log_s.sum(-1, keepdims=True)

    def generative_fun(params, inputs, **kwargs):
      scale_params, translate_params = params

      masked_inputs = inputs * mask
      log_s = scale_apply_fun(scale_params, masked_inputs) * (1 - mask)
      t = translate_apply_fun(translate_params, masked_inputs) * (1 - mask)
      s = np.exp(-log_s)

      return (inputs - t) * s, log_s.sum(-1, keepdims=True)

    return (scale_params, translate_params), normalizing_fun, generative_fun
  return init_fun


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
  if mask_type == 'input':
    in_degrees = np.arange(in_features) % in_flow_features
  else:
    in_degrees = np.arange(in_features) % (in_flow_features - 1)

  if mask_type == 'output':
    out_degrees = np.arange(out_features) % in_flow_features - 1
  else:
    out_degrees = np.arange(out_features) % (in_flow_features - 1)

  mask = np.transpose(np.expand_dims(out_degrees, -1) >= np.expand_dims(in_degrees, 0)).astype(np.float32)
  return mask


def MaskedDense(out_dim, mask, W_init=glorot_normal(), b_init=normal()):
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    k1, k2 = random.split(rng)
    W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
    return output_shape, (W, b)

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
    A new layer, i.e. a (params, normalizing_fun, generative_fun) triplet
  """
  def init_fun(rng, input_shape):
    num_hidden = 64
    num_inputs = input_shape[-1]

    input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
    hidden_mask = get_mask(num_hidden, num_hidden, num_inputs, mask_type=None)
    output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')

    joiner_init_fun, joiner_apply_fun = MaskedDense(num_hidden, input_mask)
    _, joiner_params = joiner_init_fun(rng, input_shape)

    trunk_init_fun, trunk_apply_fun = stax.serial(
      Relu,
      MaskedDense(num_hidden, hidden_mask),
      Relu,
      MaskedDense(num_inputs * 2, output_mask)
    )
    _, trunk_params = trunk_init_fun(rng, (num_hidden,))

    def normalizing_fun(params, inputs, **kwargs):
      joiner_params, trunk_params = params

      h = joiner_apply_fun(joiner_params, inputs)
      m, a = trunk_apply_fun(trunk_params, h).split(2, 1)
      u = (inputs - m) * np.exp(-a)

      return u, -a.sum(-1, keepdims=True)

    def generative_fun(params, inputs, **kwargs):
      joiner_params, trunk_params = params

      x = np.zeros_like(inputs)
      for i_col in range(inputs.shape[1]):
        h = joiner_apply_fun(joiner_params, x)
        m, a = trunk_apply_fun(trunk_params, h).split(2, 1)
        # x[:, i_col] = inputs[:, i_col] * np.exp(a[:, i_col]) + m[:, i_col]
        x = jax.ops.index_update(x, jax.ops.index[:, i_col], inputs[:, i_col] * np.exp(a[:, i_col]) + m[:, i_col])

      return x, -a.sum(-1, keepdims=True)

    return (joiner_params, trunk_params), normalizing_fun, generative_fun
  return init_fun


def serial(*init_funs):
  def init_fun(rng, input_shape):
    params, normalizing_funs, generative_funs = [], [], []
    for init_fun in init_funs:
      rng, layer_rng = random.split(rng)
      param, normalizing_fun, generative_fun = init_fun(layer_rng, input_shape)

      params.append(param)
      normalizing_funs.append(normalizing_fun)
      generative_funs.append(generative_fun)

    def normalizing_fun(params, inputs, **kwargs):
      rng = kwargs.pop('rng', None)
      rngs = random.split(rng, len(init_funs)) if rng is not None else (None,) * len(init_funs)
      logdets = None
      for fun, param, rng in zip(normalizing_funs, params, rngs):
        inputs, logdet = fun(param, inputs, rng=rng, **kwargs)
        if logdets is None:
            logdets = logdet
        else:
            logdets += logdet
      return inputs, logdets

    def generative_fun(params, inputs, **kwargs):
      rng = kwargs.pop('rng', None)
      rngs = random.split(rng, len(init_funs)) if rng is not None else (None,) * len(init_funs)
      logdets = None
      for fun, param, rng in reversed(list(zip(generative_funs, params, rngs))):
        inputs, logdet = fun(param, inputs, rng=rng, **kwargs)
        if logdets is None:
            logdets = logdet
        else:
            logdets += logdet
      return inputs, logdets

    return params, normalizing_fun, generative_fun
  return init_fun


def net(rng, input_shape, hidden_dim=64, act=Relu):
  init_fun, apply_fun = stax.serial(
    Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
    act,
    Dense(hidden_dim, W_init=orthogonal(), b_init=zeros),
    act,
    Dense(input_shape[-1], W_init=orthogonal(), b_init=zeros),
  )
  _, params = init_fun(rng, input_shape)
  return (params, apply_fun)


def mask(input_shape):
  mask = onp.zeros(input_shape)
  mask[::2] = 1.
  return mask


def log_probs(params, normalizing_fun, inputs):
  u, log_jacob = normalizing_fun(params, inputs)
  log_probs = (-.5 * (u ** 2.) - .5 * np.log(2 * math.pi)).sum(-1, keepdims=True)
  return (log_probs + log_jacob).sum(-1, keepdims=True)


def NLL(params, normalizing_fun, inputs):
  return -log_probs(params, normalizing_fun, inputs).mean()


def GaussianPrior():
  def pdf(inputs):
    return (-.5 * (inputs ** 2.) - .5 * np.log(2 * math.pi)).sum(-1, keepdims=True)

  def sample(input_shape, num_samples=1):
    return npr.normal(0., 1., (num_samples,) + input_shape)

  return pdf, sampler


def Flow(transformation_init, rng, input_shape, prior=GaussianPrior()):
  params, normalizing_fun, _  = transformation_init_fun(rng, input_shape)
  pdf, sampler = prior

  def log_prob(inputs):
    u, log_jacob = normalizing_fun(params, inputs)
    log_probs = pdf(u)
    return (log_probs + log_jacob).sum(-1, keepdims=True)

  def sample(num_samples):
    return sampler(input_shape, num_samples)

  return params, log_prob, sample

