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

"""Tests for evaluate video embedding."""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from brave.datasets import datasets
from brave.datasets import fixtures
from brave.evaluate import evaluate_video_embedding


class EvaluateVideoEmbeddingTest(parameterized.TestCase):

  def test_evaluate_embedding(self):
    with tempfile.TemporaryDirectory() as fixture_dir:
      shards = fixtures.write_tf_record_dataset_fixture(fixture_dir)
      rng = jax.random.PRNGKey(0)

      def fake_embedding(view, is_training):
        del is_training
        video = view.video
        chex.assert_rank(video, 5)  # B, T, H, W, C
        flat_video = jnp.reshape(video, (video.shape[0], -1))
        feats = flat_video[..., :16]
        chex.assert_shape(feats, (None, 16))

        return hk.Linear(2048)(feats)

      fake_embedding_fn = hk.transform(fake_embedding)
      view = datasets.View(
          labels=None,
          audio=None,
          video=np.zeros((1, 2, 8, 8, 3)),
      )
      params = fake_embedding_fn.init(rng, view, True)

      def embedding_fn(view):
        return fake_embedding_fn.apply(params, rng, view, False)

      train_shards = shards
      test_shards = shards
      config = evaluate_video_embedding.VideoConfig(
          num_frames=2,
          image_size=8,
          video_step=1,
      )

      results = evaluate_video_embedding.evaluate_video_embedding(
          train_shards,
          test_shards,
          embedding_fn,
          config,
          svm_regularization=1.0)

      self.assertLessEqual(results.test.top_one_accuracy, 1.0)
      self.assertGreaterEqual(results.test.top_one_accuracy, 0.0)


if __name__ == '__main__':
  absltest.main()
