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

"""Tests for sampling."""

from absl.testing import absltest
from absl.testing import parameterized

from brave.datasets import sampling


class SamplingTest(parameterized.TestCase):

  @parameterized.named_parameters([
      {
          'testcase_name': 'single_item',
          'sequence_length': 10,
          'num_samples': 1,
          'num_frames_per_sample': 1,
          'step': 1,
          'expected': [(0, 1, 1)]
      },
      {
          'testcase_name': 'two_items',
          'sequence_length': 10,
          'num_samples': 2,
          'num_frames_per_sample': 1,
          'step': 1,
          'expected': [(0, 1, 1), (9, 10, 1)]
      },
      {
          'testcase_name': 'five_items',
          'sequence_length': 5,
          'num_samples': 5,
          'num_frames_per_sample': 1,
          'step': 1,
          'expected': [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)]
      },
      {
          'testcase_name': 'overlapping_items',
          'sequence_length': 5,
          'num_samples': 5,
          'num_frames_per_sample': 3,
          'step': 1,
          'expected': [(0, 3, 1), (0, 3, 1), (1, 4, 1), (1, 4, 1), (2, 5, 1)]
      },
      {
          'testcase_name': 'overlapping_with_step',
          'sequence_length': 12,
          'num_samples': 3,
          'num_frames_per_sample': 3,
          'step': 2,
          'expected': [(0, 5, 2), (3, 8, 2), (7, 12, 2)]
      },
      {
          'testcase_name': 'shortest_possible_sequence',
          'sequence_length': 12,
          'num_samples': 3,
          'num_frames_per_sample': 12,
          'step': 1,
          'expected': [(0, 12, 1), (0, 12, 1), (0, 12, 1)]
      },
      {
          'testcase_name': 'shortest_possible_sequence_with_step',
          'sequence_length': 4,
          'num_samples': 3,
          'num_frames_per_sample': 2,
          'step': 3,
          'expected': [(0, 4, 3), (0, 4, 3), (0, 4, 3)]
      },
  ])
  def test_compute_linearly_spaced_sample_indices(self, sequence_length,
                                                  num_samples,
                                                  num_frames_per_sample, step,
                                                  expected):
    result = sampling.compute_linearly_spaced_sample_indices(
        sequence_length, num_samples, num_frames_per_sample, step)
    result = [tuple(v) for v in result]
    self.assertEqual(expected, result)


if __name__ == '__main__':
  absltest.main()
