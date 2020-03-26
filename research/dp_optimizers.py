from jax import grad
from jax import jit
from jax import partial
from jax import random
from jax import tree_util
from jax import vmap
from jax.experimental import optimizers
from jax.experimental import stax
from jax.lax import stop_gradient
import jax.numpy as np
from examples import datasets
import numpy.random as npr


def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier, batch_size):
  """Return differentially private gradients for params, evaluated on batch."""

  def _clipped_grad(params, single_example_batch):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = grad(loss)(params, single_example_batch)

    nonempty_grads, tree_def = tree_util.tree_flatten(grads)
    total_grad_norm = np.linalg.norm(
        [np.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = stop_gradient(np.amax((total_grad_norm / l2_norm_clip, 1.)))
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)

  px_clipped_grad_fn = vmap(partial(_clipped_grad, params))
  std_dev = l2_norm_clip * noise_multiplier
  noise_ = lambda n: n + std_dev * random.normal(rng, n.shape)
  normalize_ = lambda n: n / float(batch_size)
  tree_map = tree_util.tree_map
  sum_ = lambda n: np.sum(n, 0)  # aggregate
  aggregated_clipped_grads = tree_map(sum_, px_clipped_grad_fn(batch))
  noised_aggregated_clipped_grads = tree_map(noise_, aggregated_clipped_grads)
  normalized_noised_aggregated_clipped_grads = (
      tree_map(normalize_, noised_aggregated_clipped_grads)
  )
  return normalized_noised_aggregated_clipped_grads


def shape_as_image(images, labels, dummy_dim=False):
  target_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
  return np.reshape(images, target_shape), labels


def compute_epsilon(steps, num_examples=60000, target_delta=1e-5):
  if num_examples * target_delta > 1.:
    warnings.warn('Your delta might be too high.')
  q = FLAGS.batch_size / float(num_examples)
  orders = list(np.linspace(1.1, 10.9, 99)) + range(11, 64)
  rdp_const = compute_rdp(q, FLAGS.noise_multiplier, steps, orders)
  eps, _, _ = get_privacy_spent(orders, rdp_const, target_delta=target_delta)
  return eps


opt_init, opt_update, get_params = optimizers.sgd(FLAGS.learning_rate)

@jit
def update(_, i, opt_state, batch):
params = get_params(opt_state)
return opt_update(i, grad(loss)(params, batch), opt_state)

@jit
def private_update(rng, i, opt_state, batch):
params = get_params(opt_state)
rng = random.fold_in(rng, i)  # get new key for new random numbers
return opt_update(
    i,
    private_grad(params, batch, rng, FLAGS.l2_norm_clip, FLAGS.noise_multiplier, FLAGS.batch_size),
    opt_state
)

_, init_params = init_random_params(key, (-1, 28, 28, 1))
opt_state = opt_init(init_params)
itercount = itertools.count()

steps_per_epoch = 60000 // FLAGS.batch_size
print('\nStarting training...')
for epoch in range(1, FLAGS.epochs + 1):
start_time = time.time()
# pylint: disable=no-value-for-parameter
for _ in range(num_batches):
  if FLAGS.dpsgd:
    opt_state = \
        private_update(
            key, next(itercount), opt_state,
            shape_as_image(*next(batches), dummy_dim=True))
  else:
    opt_state = update(
        key, next(itercount), opt_state, shape_as_image(*next(batches)))

# pylint: enable=no-value-for-parameter
epoch_time = time.time() - start_time
print('Epoch {} in {:0.2f} sec'.format(epoch, epoch_time))

# evaluate test accuracy
params = get_params(opt_state)
test_acc = accuracy(params, shape_as_image(test_images, test_labels))
test_loss = loss(params, shape_as_image(test_images, test_labels))
print('Test set loss, accuracy (%): ({:.2f}, {:.2f})'.format(test_loss, 100 * test_acc))

# determine privacy loss so far
if FLAGS.dpsgd:
  delta = 1e-5
  num_examples = 60000
  eps = compute_epsilon(epoch * steps_per_epoch, num_examples, delta)
  print('For delta={:.0e}, the current epsilon is: {:.2f}'.format(delta, eps))
else:
  print('Trained with vanilla non-private SGD optimizer')

