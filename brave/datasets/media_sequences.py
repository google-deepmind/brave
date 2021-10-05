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

"""A package for reading and materializing encoded data into memory."""

import enum
import functools
from typing import Any, Callable, Dict, Optional, Sequence

import chex
import tensorflow as tf

MAXIMUM_DATA_FETCH_CONCURRENT_REQUESTS = 8
DEFAULT_BLOCK_LENGTH = 8
DEFAULT_SHUFFLE_BUFFER_SIZE = 32

# Note(rhemsley): We are fortunate that the datasets we currently use follow the
# same conventions for naming these features. We may have to support different
# feature names in future.
DEFAULT_LABEL_FEATURE_NAME = 'clip/label/index'
DEFAULT_IMAGE_FEATURE_NAME = 'image/encoded'
DEFAULT_AUDIO_FEATURE_NAME = 'WAVEFORM/feature/floats'

# A ShardReaderFn takes a path and returns a tf.data.Dataset. The dataset
# must be an iterable over serialized tf.train.SequenceExample protos in
# tensors.
ShardReaderFn = Callable[[str], tf.data.Dataset]


class FeatureKind(enum.Enum):
  VIDEO = enum.auto()
  LABELS = enum.auto()
  AUDIO = enum.auto()


@chex.dataclass
class EncodedSequence:
  """Encoded sequences contain selected fields read from MediaSequence protos.

  We make use of an optimized proto reader to construct these objects, which
  is why we do not use the proto message directly.

  Attributes:
    jpeg_encoded_images: An optional tensor of shape (T,) containing jpeg
      encoded strings. Each entry in the tensor corresponds to an (ordered)
      frame in a video.
    audio: Raw audio encoded as a waveform, with rank-1 and of shape (T,). For
      example, for 10 seconds of audio sampled at 48Khz, the shape would be
      (480000,).
    labels: An optional tensor of integer class indices of shape (N,), where N
      is the number of labels associated with this sequence.
  """
  jpeg_encoded_images: Optional[tf.Tensor]
  audio: Optional[tf.Tensor]
  labels: Optional[tf.Tensor]

  def __repr__(self):
    features = []

    if self.jpeg_encoded_images is not None:
      features.append(f'jpeg_encoded_images: {self.jpeg_encoded_images.shape}')
    if self.labels is not None:
      features.append(f'labels: {self.labels.shape}')
    if self.audio is not None:
      features.append(f'audio: {self.audio.shape}')

    return '<EncodedSequence ' + ', '.join(features) + '>'


def tf_record_shard_reader(path: str) -> tf.data.Dataset:
  """By default, we assume that the data can be read using from TFRecords."""
  return tf.data.TFRecordDataset(path)


def media_sequence_dataset(
    shards: Sequence[str],
    features: Sequence[FeatureKind],
    *,
    shuffle: bool = False,
    shard_reader: ShardReaderFn = tf.data.TFRecordDataset) -> tf.data.Dataset:
  """Returns a tensorflow dataset that iterates over encoded media sequences.

  Uses jax.process_count() and jax.process_index() to shard the data across
  the different active processes.

  Args:
    shards: The paths to the shards to read.
    features: The features to read from the encoded protobuf.
    shuffle: Whether or not to shuffle the data.
    shard_reader: A function mapping a path from the sequence of shards and
      return a tf.data.Dataset over serializedtf.train.SequenceExample protos.
      Defaults to a reader for `TFRecordDataset`.

  Returns:
    A tf.data.Dataset containing objects of type EncodedSequence.
  """

  # Create a dataset that iterates over the shard paths.
  ds = tf.data.Dataset.from_tensor_slices(shards)

  # Shuffling the shards is an efficient way to shuffle the dataset at
  # a coarse level of granularity.
  if shuffle:
    ds = ds.shuffle(len(shards), seed=0)

  # We map the shard reader function across the shards and interleave the
  # results in parallel, resulting in parallel reads to the shards that are
  # combined into one sequential dataset.
  # According to the docs, the cycle_length becomes the maximum concurrent
  # fetches.
  ds = ds.interleave(
      shard_reader,
      cycle_length=MAXIMUM_DATA_FETCH_CONCURRENT_REQUESTS,
      block_length=DEFAULT_BLOCK_LENGTH,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=not shuffle)

  # Deserialize only the required features from the protobuf.
  ds = ds.map(functools.partial(_extract_features_from_pbf, features=features))

  # Despite shuffling the shards, we still need to shuffle within the shards.
  if shuffle:
    ds = ds.shuffle(buffer_size=DEFAULT_SHUFFLE_BUFFER_SIZE)

  return ds


def extend_sequence(sequence: EncodedSequence,
                    min_video_frames: int) -> EncodedSequence:
  """Extend a sequence until it contains at least the given number of frames.

  Args:
    sequence: The sequence under consideration
    min_video_frames: The minimum number of video frames required.

  Returns:
    A sequence containing at least the minimum number of frames.
  """
  num_frames = tf.shape(sequence.jpeg_encoded_images)[0]
  needed_repeats = tf.math.ceil(min_video_frames / num_frames)

  return tf.cond(
      tf.greater(needed_repeats, 1.0), lambda: repeat(sequence, needed_repeats),
      lambda: sequence)


def repeat(sequence: EncodedSequence, n: int) -> EncodedSequence:
  """Extend a sequence by repeating it n times.

  This is useful to extend unusually short sequences to allow sampling longer
  time ranges.

  Args:
    sequence: The sequence to repeat.
    n: The number of times to loop the input seuquence.

  Returns:
    A new sequence that is n times longer than the input sequence.
  """

  result = EncodedSequence(jpeg_encoded_images=None, audio=None, labels=None)

  if sequence.jpeg_encoded_images is not None:
    result.jpeg_encoded_images = tf.tile(
        sequence.jpeg_encoded_images, multiples=(n,))

  if sequence.audio is not None:
    result.audio = tf.tile(sequence.audio, multiples=(n,))

  if sequence.labels is not None:
    result.labels = sequence.labels

  return result


def _extract_features_from_pbf(
    buffer: tf.Tensor, features: Sequence[FeatureKind]) -> EncodedSequence:
  """Read specific features from a media sequence proto.

  Args:
    buffer: A serialized tf.train.SequenceExample proto.
    features: The features that should be read into the resulting object.

  Returns:
    An EncodedSequence object containing the requested features.
  """

  features_dct = _build_features_dct(features)
  context_features_dct = _build_context_features_dct(features)
  context_dct, dct = tf.io.parse_single_sequence_example(
      buffer, context_features_dct, features_dct)

  result = EncodedSequence(jpeg_encoded_images=None, audio=None, labels=None)

  if DEFAULT_IMAGE_FEATURE_NAME in dct:
    result.jpeg_encoded_images = tf.identity(dct[DEFAULT_IMAGE_FEATURE_NAME])

  if DEFAULT_AUDIO_FEATURE_NAME in dct:
    audio = tf.sparse.to_dense(dct[DEFAULT_AUDIO_FEATURE_NAME])

    # The audio is stored on the wire as (1, <n samples>).
    tf.assert_rank(audio, 2)
    result.audio = audio[0]

  if DEFAULT_LABEL_FEATURE_NAME in context_dct:
    result.labels = tf.sparse.to_dense(context_dct[DEFAULT_LABEL_FEATURE_NAME])

  return result


def _build_context_features_dct(
    features: Sequence[FeatureKind]) -> Dict[str, Any]:
  dct = {}
  for feature in features:
    if feature is FeatureKind.LABELS:
      dct[DEFAULT_LABEL_FEATURE_NAME] = tf.io.VarLenFeature(dtype=tf.int64)
  return dct


def _build_features_dct(features: Sequence[FeatureKind]) -> Dict[str, Any]:
  """Build the input dictionary required by parse_single_sequence_example.

  Due to optimizations in parse_single_sequence_example, we need to specify
  additional information about the way the fields should be loaded.

  Args:
    features: The features to load

  Returns:
    Type information used to construct the tensors for the encoded sequence
    objects.
  """
  dct = {}

  for feature in features:
    if feature is FeatureKind.VIDEO:
      dct[DEFAULT_IMAGE_FEATURE_NAME] = tf.io.FixedLenSequenceFeature(
          (), dtype=tf.string)
    if feature is FeatureKind.AUDIO:
      dct[DEFAULT_AUDIO_FEATURE_NAME] = tf.io.VarLenFeature(dtype=tf.float32)

  return dct
