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

"""Tests for eval datasets."""

import tempfile

from absl.testing import absltest

from brave.datasets import fixtures
from brave.evaluate import eval_datasets


class DatasetsTest(absltest.TestCase):

  def test_multiview_sampling_dataset(self):
    with tempfile.TemporaryDirectory() as fixture_dir:
      shards = fixtures.write_tf_record_dataset_fixture(fixture_dir)

      ds = eval_datasets.multiple_crop_dataset(
          shards,
          num_temporal_crops=3,
          num_spatial_crops=2,
          num_video_frames=4,
          video_step=2,
          initial_resize=224,
          center_crop_size=128,
          shuffle=False)

      seen_batches = 0
      for batch in ds:
        self.assertEqual(batch.views['default'].video.shape, (4, 128, 128, 3))
        self.assertEqual(batch.views['default'].labels.shape, (1,))
        seen_batches += 1

      self.assertEqual(seen_batches, 18)

  def test_random_sampling_dataset(self):
    with tempfile.TemporaryDirectory() as fixture_dir:
      shards = fixtures.write_tf_record_dataset_fixture(fixture_dir)
      ds = eval_datasets.random_sampling_dataset(
          shards,
          num_video_frames=3,
          video_step=2,
          image_size=128,
          min_crop_window_area=0.4,
          max_crop_window_area=0.6,
          min_crop_window_aspect_ratio=0.3,
          max_crop_window_aspect_ratio=0.6)

      for batch in ds:
        self.assertEqual(batch.views['default'].video.shape, (3, 128, 128, 3))


if __name__ == '__main__':
  absltest.main()
