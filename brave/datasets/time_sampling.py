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

"""Logic for sampling in the temporal dimension."""

from typing import NamedTuple, Optional, Union
import tensorflow as tf

from brave.datasets import media_sequences
from brave.datasets import sampling

Scalar = Union[int, tf.Tensor]


class SampleResult(NamedTuple):
  sequence: media_sequences.EncodedSequence
  indices: sampling.Indices


def random_sample_sequence_using_video(
    num_video_frames: Scalar,
    video_frame_step: Scalar,
    sequence: media_sequences.EncodedSequence,
    seed: Optional[int] = None,
    sample_start_index: Scalar = 0,
    sample_end_index: Scalar = -1,
    override_num_audio_samples: Optional[int] = None,
) -> SampleResult:
  """Randomly sample a sub-sequence using video to sync.

  Args:
    num_video_frames: The number of video frames to return.
    video_frame_step: The step between frames sampled from the sequence.
    sequence: The sequence to sample from.
    seed: A seed to set to introduce determinism.
    sample_start_index: The returned must start at or after this value.
    sample_end_index: The returned must end before this value (can be negative).
      Thus the returned indices must fall within the range given by
      [sample_start_index, sample_end_index).
    override_num_audio_samples: If set, and audio is present, then the length of
      the sampled audio will be set to this value. This is useful to avoid
      rounding and sync. errors where we want the audio tensor to have a
      specific shape.

  Returns:
    A new sequence, where the video has been sampled to have the given
    number of frames and the given step. All other fields present in the
    sequence are also sampled proportionally to the same time range.
  """

  # Change negative values of sample_end_index into a true index value.
  total_frames = tf.shape(sequence.jpeg_encoded_images)[0]
  sample_end_index = tf.cond(
      tf.less(sample_end_index, 0), lambda: total_frames + sample_end_index + 1,
      lambda: sample_end_index)

  indices = sampling.random_sample(
      start_index=sample_start_index,
      end_index=sample_end_index,
      sample_length=num_video_frames,
      step=video_frame_step,
      seed=seed)

  new_sequence = get_subsequence_by_video_indices(
      sequence, indices, override_num_audio_samples=override_num_audio_samples)
  return SampleResult(new_sequence, indices)


def get_subsequence_by_video_indices(
    sequence: media_sequences.EncodedSequence,
    indices: sampling.Indices,
    override_num_audio_samples: Optional[int] = None
) -> media_sequences.EncodedSequence:
  """Return a new subsequence sliced down using the video to sync time.

  Args:
    sequence: The sequence to slice using indices.
    indices: The indices to use to slice the input.
    override_num_audio_samples: If set, and audio is present, then the length of
      the sampled audio will be set to this value. This is useful to avoid
      rounding and sync. errors where we want the tensors to have a specific
      shape.

  Returns:
    A new sequence, sliced using the given indices (which are specifically
    for the `jpeg_encoded_images` part of the input data.

    The other components of the input sequence will be sliced synchronously
    to the same sub-region (although with no 'step' applied).

    The labels are never sliced, and are always kept the same.
  """

  result = media_sequences.EncodedSequence(
      jpeg_encoded_images=None, audio=None, labels=sequence.labels)

  result.jpeg_encoded_images = sequence.jpeg_encoded_images[
      indices.start_index:indices.end_index:indices.step]

  if sequence.labels is not None:
    result.labels = sequence.labels

  if sequence.audio is not None:
    audio = sequence.audio
    video_length = tf.shape(sequence.jpeg_encoded_images)[0]
    audio_length = tf.shape(sequence.audio)[0]
    audio_indices = synced_indices(indices, video_length, audio_length)

    if override_num_audio_samples is not None:
      audio_indices = sampling.Indices(
          audio_indices.start_index,
          audio_indices.start_index + override_num_audio_samples,
          audio_indices.step)

      audio_length = tf.shape(sequence.audio)[-1]
      if audio_length < audio_indices.end_index:
        padding = tf.zeros((audio_indices.end_index - audio_length,))
        audio = tf.concat([audio, padding], axis=-1)

    result.audio = audio[audio_indices.start_index:audio_indices
                         .end_index:audio_indices.step]

  return result


def synced_indices(indices: sampling.Indices,
                   old_length: Scalar,
                   new_length: Scalar,
                   new_step: Scalar = 1) -> sampling.Indices:
  """Move indices in one array to equivalent indices in another.

  Args:
    indices: The indices to resample/
    old_length: The length of the array the indices came from.
    new_length: The length of the array we would like to sample in.
    new_step: The step value to return with the returned sequence.

  Returns:
    Indices, modified so that they sample the same region in another array.
  """

  length = (indices.end_index - indices.start_index)
  ratio = tf.cast(new_length / old_length, tf.float32)
  start = tf.cast(tf.cast(indices.start_index, tf.float32) * ratio, tf.int32)
  end = tf.cast((tf.cast(indices.start_index + length, tf.float32)) * ratio,
                tf.int32)

  start = tf.maximum(0, start)
  end = tf.minimum(new_length, end)

  return sampling.Indices(
      start_index=start,
      end_index=end,
      step=new_step,
  )
