# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library for computing privacy values for DP-SGD."""


import math
import numpy as np
from .rdp_accountant import compute_rdp, compute_rdp_sample_without_replacement, get_privacy_spent
import sys


def compute_eps_poisson(iterations, noise_multiplier, n, batch_size, delta):
  """Compute epsilon based on the given hyperparameters."""
  q = batch_size / n
  if q > 1:
    raise Exception('n must be larger than the batch size.')
  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] + list(range(5, 64)) + [128, 256, 512])

  rdp = compute_rdp(q, noise_multiplier, iterations, orders)
  eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)

  return eps


def compute_eps_uniform(iterations, noise_multiplier, n, batch_size, delta):
  q = batch_size / n
  if q > 1:
    raise Exception('n must be larger than the batch size.')
  orders = np.linspace(2, 100, 99)

  rdp = compute_rdp_sample_without_replacement(q, noise_multiplier, iterations, orders)
  eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)

  return eps
