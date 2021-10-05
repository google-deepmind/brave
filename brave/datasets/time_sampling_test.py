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

"""Tests for time sampling."""

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from brave.datasets import media_sequences
from brave.datasets import sampling
from brave.datasets import time_sampling


class TimeSamplingTest(parameterized.TestCase):

  def test_video_sampling_overruns_region(self):
    sequence = _sequence_fixture()

    # There is only one way to sample
    result = time_sampling.random_sample_sequence_using_video(
        num_video_frames=2, video_frame_step=3, sequence=sequence, seed=5)
    expected_images = tf.constant(['abc', 'jkl'])
    tf.assert_equal(expected_images, result.sequence.jpeg_encoded_images)
    tf.assert_equal(0, result.indices.start_index)
    tf.assert_equal(4, result.indices.end_index)

    expected_audio = tf.constant([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    tf.assert_equal(expected_audio, result.sequence.audio)

  def test_impossible_time_sampling(self):
    sequence = _sequence_fixture()
    with self.assertRaises(tf.errors.InvalidArgumentError):
      time_sampling.random_sample_sequence_using_video(
          num_video_frames=3, video_frame_step=3, sequence=sequence, seed=5)

  def test_all_values_are_sampled(self):
    sequence = _sequence_fixture()
    samples = set()

    for _ in range(100):
      result = time_sampling.random_sample_sequence_using_video(
          num_video_frames=1, video_frame_step=1, sequence=sequence)

      vs = tuple(result.sequence.jpeg_encoded_images.numpy().tolist())
      samples.add(vs)

    expected = set([(b'abc',), (b'def',), (b'ghi',), (b'jkl',)])
    self.assertEqual(expected, samples)

  def test_time_sample_length_1(self):
    sequence = _sequence_fixture()
    samples = set()

    for _ in range(100):
      result = time_sampling.random_sample_sequence_using_video(
          num_video_frames=1,
          video_frame_step=1,
          sequence=sequence,
          sample_start_index=2)

      vs = tuple(result.sequence.jpeg_encoded_images.numpy().tolist())
      samples.add(vs)

    expected = set([(b'ghi',), (b'jkl',)])
    self.assertEqual(expected, samples)

  def test_constrained_sample_ranges(self):
    sequence = _sequence_fixture()
    result = time_sampling.random_sample_sequence_using_video(
        num_video_frames=2,
        video_frame_step=1,
        sequence=sequence,
        sample_start_index=1,
        sample_end_index=3)

    # Only one sequence can satisfy these constraints.
    expected_images = tf.constant(['def', 'ghi'])
    tf.assert_equal(expected_images, result.sequence.jpeg_encoded_images)

    expected_audio = tf.constant([2.0, 3.0, 4.0, 5.0])
    tf.assert_equal(expected_audio, result.sequence.audio)

  @parameterized.parameters([
      {
          'indices': (0, 2, 1),
          'old_length': 4,
          'new_length': 4,
          'expected': (0, 2, 1),
      },
      {
          'indices': (0, 1, 1),
          'old_length': 4,
          'new_length': 8,
          'expected': (0, 2, 1),
      },
      {
          'indices': (0, 1, 1),
          'old_length': 4,
          'new_length': 8,
          'expected': (0, 2, 1),
      },
      {
          'indices': (0, 4, 3),
          'old_length': 4,
          'new_length': 4,
          'expected': (0, 4, 1),
      },
      {
          'indices': (0, 10, 4),
          'old_length': 10,
          'new_length': 5,
          'expected': (0, 5, 1),
      },
  ])
  def test_synced_indices(self, indices, old_length, new_length, expected):
    indices = sampling.Indices(*indices)
    result = time_sampling.synced_indices(indices, old_length, new_length)
    self.assertEqual(expected[0], result.start_index)
    self.assertEqual(expected[1], result.end_index)
    self.assertEqual(expected[2], result.step)

  def test_get_subsequence_by_video_indices(self):
    sequence = _sequence_fixture()
    result = time_sampling.get_subsequence_by_video_indices(
        sequence, sampling.Indices(1, 3, 1))
    expected_images = tf.constant(['def', 'ghi'])
    expected_audio = tf.constant([2.0, 3.0, 4.0, 5.0])
    tf.assert_equal(expected_audio, result.audio)
    tf.assert_equal(expected_images, result.jpeg_encoded_images)

    result = time_sampling.get_subsequence_by_video_indices(
        sequence, sampling.Indices(1, 3, 1), override_num_audio_samples=8)
    expected_images = tf.constant(['def', 'ghi'])
    expected_audio = tf.constant([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0])
    tf.assert_equal(expected_audio, result.audio)
    tf.assert_equal(expected_images, result.jpeg_encoded_images)


def _sequence_fixture():
  return media_sequences.EncodedSequence(
      jpeg_encoded_images=tf.constant(['abc', 'def', 'ghi', 'jkl']),
      audio=tf.constant([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
      labels=None)


if __name__ == '__main__':
  absltest.main()
