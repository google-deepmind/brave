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

"""Tests for media sequences."""
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from brave.datasets import fixtures
from brave.datasets import media_sequences


class MediaSequencesTest(parameterized.TestCase):

  def test_read_hmdb_51(self):
    with tempfile.TemporaryDirectory() as fixture_dir:
      self.shards = fixtures.write_tf_record_dataset_fixture(fixture_dir)

      features = [
          media_sequences.FeatureKind.VIDEO,
          media_sequences.FeatureKind.LABELS,
          media_sequences.FeatureKind.AUDIO,
      ]

      ds = media_sequences.media_sequence_dataset(self.shards, features)

      for v in ds:
        self.assertIsInstance(v, media_sequences.EncodedSequence)
        self.assertIsNotNone(v.jpeg_encoded_images)
        self.assertIsNotNone(v.audio)
        self.assertIsNotNone(v.labels)

  def test_repeat(self):
    sequence = media_sequences.EncodedSequence(
        jpeg_encoded_images=tf.constant(['abc', 'def', 'ghi', 'jkl']),
        audio=tf.zeros((480376,)),
        labels=None)

    result = media_sequences.repeat(sequence, 2)
    expected = tf.constant(
        ['abc', 'def', 'ghi', 'jkl', 'abc', 'def', 'ghi', 'jkl'])
    tf.assert_equal(expected, result.jpeg_encoded_images)

    self.assertEqual((2 * 480376,), tf.shape(result.audio))

  @parameterized.parameters([
      dict(min_num_frames=1, expected_length=4),
      dict(min_num_frames=3, expected_length=4),
      dict(min_num_frames=4, expected_length=4),
      dict(min_num_frames=5, expected_length=8),
      dict(min_num_frames=8, expected_length=8),
      dict(min_num_frames=9, expected_length=12),
      dict(min_num_frames=13, expected_length=16),
      dict(min_num_frames=16, expected_length=16),
  ])
  def test_extend_sequence(self, min_num_frames, expected_length):
    sequence = media_sequences.EncodedSequence(
        jpeg_encoded_images=tf.constant(['abc', 'def', 'ghi', 'jkl']),
        audio=tf.constant([0.1, -0.1, 0.0, 0.0]),
        labels=None)

    result = media_sequences.extend_sequence(sequence, min_num_frames)
    self.assertEqual(expected_length, result.jpeg_encoded_images.shape[0])
    self.assertEqual(expected_length, result.audio.shape[0])


if __name__ == '__main__':
  absltest.main()
