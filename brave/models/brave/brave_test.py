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

"""Tests for Brave."""

import copy
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jaxline import utils as jaxline_utils
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds

from brave.datasets import datasets
from brave.datasets import fixtures
from brave.models.brave import brave
from brave.training import trainer

ConfigDict = ml_collections.ConfigDict

TEST_CONFIG = ConfigDict({
    'num_frames_broad': 2,
    'image_size_broad': 1,
    'step_broad': 1,
    'include_video_in_broad_view': True,
    'include_audio_in_broad_view': True,
    'num_frames_narrow': 1,
    'image_size_narrow': 2,
    'step_narrow': 1,
    'output_dims': 4,
    'tsm_resnet_width_multiplier': 1,
    'num_spectrogram_bins': 2,
    'fft_step': 160,
    'dataset_shards': None,
    'input_video_sample_rate': 25.0,
    'input_audio_sample_rate': 16_000,
})

# The number of dimensions the backbone emits.
BACKBONE_EMBEDDING_DIMS = 2048

DEVICE_COUNT = 1
chex.set_n_cpu_devices(DEVICE_COUNT)


class BraveTest(parameterized.TestCase):

  def test_apply_embeddings_and_loss(self):
    """Test that parameters and loss can be computed for all embeddings."""

    # Using real pmap would make the test _much_ slower.
    with chex.fake_pmap(), tempfile.TemporaryDirectory() as fixture_dir:
      self.assertEqual(jax.local_device_count(), DEVICE_COUNT)
      shards = fixtures.write_tf_record_dataset_fixture(fixture_dir)

      cfg = copy.copy(TEST_CONFIG)
      cfg.unlock()
      cfg.dataset_shards = shards
      cfg.lock()

      cfg = brave.BraveConfig(**cfg)
      # A key with the right shape (note this should really be broadcasted
      # for init)
      broadcasted_key = jaxline_utils.bcast_local_devices(jax.random.PRNGKey(0))
      batch = brave.get_empty_minibatch(cfg)

      broad_view = batch.views['broad']
      narrow_view = batch.views['narrow']

      broad_view.video = np.random.uniform(size=[DEVICE_COUNT, 1, 2, 2, 2, 3])
      broad_view.audio = np.random.uniform(size=[DEVICE_COUNT, 1, 4, 2])
      narrow_view.video = np.random.uniform(size=[DEVICE_COUNT, 1, 1, 2, 2, 3])

      model = brave.get_model(cfg)
      params, state = jax.pmap(model.init_fn, axis_name='i')(broadcasted_key)
      embeddings = model.forward_fns

      def broad(params, state, view):
        return embeddings['broad_video'](params, state, None, view, False)

      def narrow(params, state, view):
        return embeddings['narrow_video'](params, state, None, view, False)

      broad = jax.pmap(broad)
      narrow = jax.pmap(narrow)

      f_b, _ = broad(params, state, broad_view)
      self.assertEqual(f_b.shape, (DEVICE_COUNT, 1, BACKBONE_EMBEDDING_DIMS))

      f_n, _ = narrow(params, state, narrow_view)
      self.assertEqual(f_n.shape, (DEVICE_COUNT, 1, BACKBONE_EMBEDDING_DIMS))
      loss_fn = model.loss_fn
      optimizer = optax.sgd(learning_rate=1e-3)
      opt_state = optimizer.init(jax.random.PRNGKey(0))

      update_fn = jax.pmap(
          trainer.build_update_fn(optimizer, loss_fn),
          axis_name='i',
          donate_argnums=(1, 2, 3, 4))

      key = jax.random.split(jax.random.PRNGKey(0), DEVICE_COUNT)
      updates = update_fn(key, batch, params, state, opt_state)
      metrics = updates.scalars

      self.assertIn('loss', metrics)
      loss = metrics['loss']
      self.assertEqual(loss.shape, (DEVICE_COUNT,))
      self.assertFalse(np.isnan(loss[0]))

      # The loss should be approximately equal to 1.0, due to the random
      # initialization of the network. Since the test is small, there may
      # a large variance, so we set the tolerance to be high.
      # Note also that we take the mean over the loss, even though there
      # is a pmean. This is due to fake_pmap not currently implementing
      # lax.pmean.
      self.assertAlmostEqual(1.0, np.mean(loss), delta=5.0)

  def test_table_reader(self):
    with tempfile.TemporaryDirectory() as fixture_dir:
      shards = fixtures.write_tf_record_dataset_fixture(fixture_dir)

      cfg = copy.copy(TEST_CONFIG)
      cfg.unlock()
      cfg.dataset_shards = shards
      cfg.lock()

      cfg = brave.BraveConfig(**cfg)
      dataset = brave._train_dataset_builder(cfg)
      ds = dataset()
      ds = ds.batch(2)
      ds = tfds.as_numpy(ds)

      for batch in ds:
        self.assertIsInstance(batch, datasets.MiniBatch)
        self.assertIn('narrow', batch.views)
        self.assertIn('broad', batch.views)
        break

  def test_avoid_nan_in_loss(self):
    """Test that degenerate points do not result in NaN loss values."""
    x = np.array([[1, 2, 3]])
    y = np.array([[0, 0, 0]])
    loss = brave._regression_loss(x, y, 1e-6)
    np.testing.assert_allclose(0.5, loss, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
