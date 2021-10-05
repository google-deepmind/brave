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

"""Tests for trainer."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from brave.datasets import datasets
from brave.training import trainer

DEVICE_COUNT = 1
chex.set_n_cpu_devices(DEVICE_COUNT)


class TrainerTest(parameterized.TestCase):

  def test_train_step(self):
    self.assertEqual(DEVICE_COUNT, jax.device_count())

    rng = jax.random.PRNGKey(0)

    def forward(x):
      return hk.nets.MLP([10, 10, 1])(x)

    def loss(batch):
      # Something that vaguely looks like a trainable embedding, but isn't.
      return jnp.mean(forward(batch.views['default'].video))

    transformed_loss_fn = hk.transform_with_state(loss)

    fake_batch = datasets.MiniBatch(
        views={
            'default':
                datasets.View(
                    video=jnp.zeros([DEVICE_COUNT, 8, 32, 32, 3]),
                    audio=None,
                    labels=None)
        })

    keys = jnp.broadcast_to(rng, (DEVICE_COUNT,) + rng.shape)
    params, state = jax.pmap(transformed_loss_fn.init)(keys, fake_batch)
    optimizer = optax.sgd(1e-4)
    opt_state = jax.pmap(optimizer.init)(params)

    # Check that the parameters are initialized to the same value.
    jax.tree_map(lambda x: np.testing.assert_allclose(x[0], x[1]), params)

    def loss_fn(params, state, rng, batch):
      loss, state = transformed_loss_fn.apply(params, state, rng, batch)
      scalars = {'loss': loss}
      return loss, (state, scalars)

    update_fn = jax.pmap(
        trainer.build_update_fn(optimizer, loss_fn), axis_name='i')

    # A smoke test to ensure that the updates are successfully computed.
    result = update_fn(keys, fake_batch, params, state, opt_state)

    # Check the parameters agree across the devices.
    jax.tree_map(lambda x: np.testing.assert_allclose(x[0], x[1]),
                 result.params)

    self.assertEqual((DEVICE_COUNT,), result.scalars['loss'].shape)

    # Due to pmean, all of the scalars should have the same value.
    loss_scalars = result.scalars['loss']
    self.assertTrue(np.all(loss_scalars == loss_scalars[0]))


if __name__ == '__main__':
  absltest.main()
