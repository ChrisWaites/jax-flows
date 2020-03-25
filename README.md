<div align="center">
    <h1>🌊 Normalizing Flows in JAX 🌊</h1>
</div>

### Example Usage

```python
import flax
from jax import random

rng = random.PRNGKey(0)
input_shape = ...

params, log_prob, sample = flax.Flow(
    flax.serial(
        flax.MADE(),
        flax.BatchNorm(),
        flax.Reverse(),
        flax.MADE(),
        flax.BatchNorm(),
        flax.Reverse(),
    ),
    flax.Gaussian(),
    rng, input_shape,
)

NLL = lambda params, inputs: -log_prob(params, inputs).mean()

@jit
def step(i, opt_state, inputs):
    params = get_params(opt_state)
    return opt_update(i, grad(NLL)(params, inputs), opt_state)

batch_size = 32
itercount = itertools.count()
for epoch in range(num_epochs):
    npr.shuffle(X)
    for batch_index in range(0, len(X), batch_size):
        opt_state = step(next(itercount), opt_state, X[batch_index:batch_index+batch_size])

optimized_params = get_params(opt_state)
```

### Layers

- RealNVP
- GLOW
- MADE
- MAF
