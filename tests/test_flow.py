import unittest

from jax import random

import flax


class TestTransformations(unittest.TestCase):
    def test_flows(self):
        input_shape = (3,)
        rng = random.PRNGKey(0)

        params, log_prob, sample = flax.Flow(
            rng, input_shape, flax.serial(flax.MADE(), flax.Reverse(), flax.MADE(), flax.Reverse(),), flax.Normal(),
        )
