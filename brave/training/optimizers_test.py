# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test the optimizer."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from brave.training import optimizers


class OptimizersTest(parameterized.TestCase):

  def test_scale_by_module_name(self):

    def forward(x):
      return hk.Sequential(
          [hk.Linear(10, name='linear_1'),
           hk.Linear(10, name='linear_2')])(
               x)

    forward_fn = hk.without_apply_rng(hk.transform(forward))
    params = forward_fn.init(jax.random.PRNGKey(0), jnp.zeros([1]))

    scaler = optimizers._scale_by_module_name([(r'^.*linear_1.*$', 10.0)])

    state = scaler.init(None)
    scaled_params, _ = scaler.update(params, state)

    np.testing.assert_allclose(10 * params['linear_1']['w'],
                               scaled_params['linear_1']['w'])
    np.testing.assert_allclose(params['linear_2']['w'],
                               scaled_params['linear_2']['w'])


if __name__ == '__main__':
  absltest.main()
