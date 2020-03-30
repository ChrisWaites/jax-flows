import unittest

from jax import random

import flows


class Tests(unittest.TestCase):
    def test_flows(self):
        input_shape = (3,)
        rng = random.PRNGKey(0)

        init_fun = flows.Flow(flows.Serial(flows.Reverse(), flows.Reverse()), flows.Normal())

        params, log_pdf, sample = init_fun(rng, input_shape)
