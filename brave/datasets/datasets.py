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

"""Datasets for multi-view multi-modal data."""

import functools
from typing import Callable, Dict, Optional, Sequence, Union

import chex
import tensorflow as tf

from brave.datasets import media_sequences

Array = Union[tf.Tensor, chex.Array]


@chex.dataclass
class View:
  """A view is a group of time-synchronized modes.

  Attributes:
    video: An optional tensor of shape [BATCH_DIMS..., T, H, W, C], where T is
      the time dimension and C=3 are the image channels as RGB.
    audio: An optional tensor of shape [BATCH_DIMS..., AUDIO...]. Depending on
      the representation, the audio may be rank-1 stored as a raw waveform of
      dimension (T,), or as a rank-2 spectrogram, (T, frequency_bins). The
      sample-rate is dependent on the dataset.
    labels: If available, contains the integer class labels for this view, of
      shape [BATCH_DIMS..., N], where N is the number of labels.
  """
  video: Optional[Array]
  audio: Optional[Array]
  labels: Optional[Array]

  def __repr__(self):
    # Workaround http://b/190464506
    if isinstance(self.video, (str, int)):
      return ''

    modes = []
    if self.video is not None:
      modes.append(f'video: {self.video.shape}')

    if self.audio is not None:
      modes.append(f'audio: {self.audio.shape}')

    if self.labels is not None:
      modes.append(f'labels: {self.labels.shape}')

    return f'<View {", ".join(modes)}'


@chex.dataclass
class MiniBatch:
  """Minibatches contain multimodal multi-view data.

  Attributes:
    views: A mapping from view_name to view. Each view is a time-synchronized
      slice of the underlying raw data. Any batch dimensions are contained in
      the tensors of the view themselves.
  """
  views: Dict[str, View]

  def __repr__(self):
    # Workaround http://b/190464506
    if isinstance(self.views, (str, int)):
      return ''

    views = ', '.join(f'{name}: {view}' for name, view in self.views.items())
    return f'<Batch {views}>'


ViewSamplerFn = Callable[[media_sequences.EncodedSequence],
                         Dict[str, media_sequences.EncodedSequence]]
ViewDecoderFn = Callable[[Dict[str, media_sequences.EncodedSequence]],
                         Dict[str, View]]


def multi_view_dataset(
    shards: Sequence[str],
    features: Sequence[media_sequences.FeatureKind],
    view_sampler: ViewSamplerFn,
    view_decoder: ViewDecoderFn,
    *,
    shuffle: bool = False,
    shard_reader: media_sequences.ShardReaderFn = media_sequences
    .tf_record_shard_reader,
) -> tf.data.Dataset:
  """Construct a multiview multimodal dataset.

  The dataset is constructed in three stages,

    * The specified features are read into EncodedSequence objects.
    * The view_sampler function is used to filter out only the fields that
      are needed for each view.
    * The view_decoder is used to decode the EncodedSequences into views.

  Args:
    shards: The shard paths to read the dataset from.
    features: The features to deserialize from the table into the encoded media
      sequences object.
    view_sampler: A callable taking an encoded media sequence and returning a
      dictionary of sampled media sequences, oen for each view.
    view_decoder: A callable taking a dictionary of encoded sequences per view
      and returning the decoded views.
    shuffle: Whether or not to shuffle the data.
    shard_reader: The callable used to decode shards from paths.

  Returns:
    A tfds dataset with underlying datatype `datasets.Minibatch`. Note that
    the returned dataset has no batch dimensions, so that every item in the
    dataset is a single example. Call `.batch()` on the result to group together
    examples.
  """

  ds = media_sequences.media_sequence_dataset(
      shards, features, shuffle=shuffle, shard_reader=shard_reader)

  return _multi_view_batches_from_sequences(
      ds, view_sampler, view_decoder, deterministic=not shuffle)


def _multi_view_batches_from_sequences(ds: tf.data.Dataset,
                                       view_sampler: ViewSamplerFn,
                                       view_decoder: ViewDecoderFn, *,
                                       deterministic: bool) -> tf.data.Dataset:
  """Construct batches using the view decoder."""

  ds = ds.map(view_sampler)

  ds = ds.map(
      functools.partial(_make_batch, view_decoder=view_decoder),
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=deterministic)

  return ds


def _make_batch(sequences: Dict[str, media_sequences.EncodedSequence],
                view_decoder: ViewDecoderFn) -> MiniBatch:
  return MiniBatch(views=view_decoder(sequences))
