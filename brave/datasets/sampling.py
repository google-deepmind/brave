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

"""Implement logic for sampling videos and audio."""

from typing import Sequence, NamedTuple, Union, Optional
import tensorflow as tf

Scalar = Union[int, tf.Tensor]


class Indices(NamedTuple):
  start_index: Scalar
  end_index: Scalar
  step: Scalar


def compute_linearly_spaced_sample_indices(sequence_length: int,
                                           num_samples: int,
                                           num_frames_per_sample: int,
                                           step: int) -> Sequence[Indices]:
  """Space out windows along a sequence.

  Args:
    sequence_length: The length of the sequence we are sampling from.
    num_samples: The number of samples to return from the input sequence.
    num_frames_per_sample: Each resulting sample must have this number of
      frames.
    step: The gap between frames sampled into the output.

  Returns:
    A sequence of slice indices referencing the input sequence.
    When `num_samples` is one, the returned sample always starts at zero,
    When `num_samples` is two, we return a sample starting at zero, and
    another sample starting at the end of the sequence (i.e. the last sample
    that could fit at the end).

    As `num_samples` increases, the samples are returned spaced out evenly
    between the first possible sample, and the last possible sample.

    Samples may overlap or indeed be repeated.
  """

  # Each cropped sample must have this number of frames. We repeat
  # the underlying sequence until it contains this number of frames, so that
  # we can return valid crops. Even if some of the samples may be repeats.
  sample_length = (num_frames_per_sample - 1) * step + 1

  last_sample_start = tf.cast(sequence_length - sample_length, tf.float32)
  start_indices = tf.linspace(0.0, last_sample_start, num_samples)
  start_indices = tf.cast(start_indices, tf.int32)

  indices = [
      Indices(start_indices[i], start_indices[i] + sample_length, step)
      for i in range(num_samples)
  ]

  return indices


def random_sample(start_index: Scalar,
                  end_index: Scalar,
                  sample_length: Scalar,
                  step: Scalar,
                  seed: Optional[int] = None) -> Indices:
  """Sample a range from a sequence given constraints.

  All arguments must be positive.

  Args:
    start_index: The returned sample must start at or after this index.
    end_index: The returned sample must end before this index.
    sample_length: The sample must contain this number of values.
    step: The sample must have this step in the original sequence.
    seed: A seed for the rng.

  Returns:
    Indices representing the start, end, and step in the original sequence.

  Raises:
    tf.error.InvalidArgumenError if the sample is not satisfiable - in this
    case, there are not enough elements in the sequence to return a sample.
  """
  samplable_sequence_length = end_index - start_index
  required_length = step * (sample_length - 1) + 1

  tf.debugging.assert_less_equal(required_length, samplable_sequence_length)

  max_val = samplable_sequence_length - required_length + 1
  idx = tf.random.uniform((), maxval=max_val, dtype=tf.int32, seed=seed)

  start = start_index + idx
  end = start + required_length

  return Indices(start, end, step)
