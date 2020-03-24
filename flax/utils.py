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

