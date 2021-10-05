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

"""Tests for datasets."""

import tempfile
from typing import Dict

from absl.testing import absltest
from absl.testing import parameterized

from brave.datasets import datasets
from brave.datasets import fixtures
from brave.datasets import media_sequences
from brave.datasets import time_sampling
from brave.datasets import video_sampling


class TFRecordDatasetsTest(parameterized.TestCase):

  def test_multi_view_dataset(self):
    with tempfile.TemporaryDirectory() as fixture_dir:
      self.shards = fixtures.write_tf_record_dataset_fixture(fixture_dir)

      # Create a video dataset with a single view called 'default' containing
      # the first 4 frames of each video, cropped to 128 X 128
      ds = datasets.multi_view_dataset(
          self.shards,
          features=[media_sequences.FeatureKind.VIDEO],
          view_sampler=_test_sampler,
          view_decoder=_test_decoder)

      for batch in ds:
        self.assertEqual(batch.views['default'].video.shape, (4, 128, 128, 3))


def _test_decoder(
    sequences: Dict[str, media_sequences.EncodedSequence]
) -> Dict[str, datasets.View]:
  batch_views = {}

  for view_name, sequence in sequences.items():
    # Now decode the crop window from the sequence as a video.
    video = video_sampling.decode_resize_crop_images(
        sequence.jpeg_encoded_images, initial_resize=224, center_crop_size=128)

    batch_views[view_name] = datasets.View(video=video, audio=None, labels=None)

  return batch_views


def _test_sampler(
    sequence: media_sequences.EncodedSequence
) -> Dict[str, media_sequences.EncodedSequence]:
  # Simply sample the first four frames of the sequence into a single view
  # called 'default'.
  result = time_sampling.random_sample_sequence_using_video(
      num_video_frames=4, video_frame_step=1, sequence=sequence)

  return {'default': result.sequence}


if __name__ == '__main__':
  absltest.main()
