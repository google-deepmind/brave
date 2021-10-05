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

"""Implement the Brave model."""

import copy
import functools
from typing import Callable, Dict, Optional, Sequence, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from brave.datasets import augmentations
from brave.datasets import datasets
from brave.datasets import media_sequences
from brave.datasets import spectrograms
from brave.datasets import time_sampling
from brave.datasets import video_sampling
from brave.models import embedding_model
from brave.models.brave import modules

FAKE_AUDIO_LENGTH = 100

EmbeddingFn = Callable[[hk.Params, hk.State, chex.PRNGKey, datasets.View, bool],
                       Tuple[chex.Array, chex.Array]]

PredictorFn = Callable[[hk.Params, hk.State, chex.PRNGKey, chex.Array, bool],
                       Tuple[chex.Array, chex.Array]]


@chex.dataclass
class ParameterizedFns:
  broad_video_embedding: EmbeddingFn
  broad_audio_embedding: EmbeddingFn
  narrow_video_embedding: EmbeddingFn
  b_video_to_n_video: PredictorFn
  n_video_to_b_video: PredictorFn
  b_audio_to_n_video: PredictorFn
  n_video_to_b_audio: PredictorFn


@chex.dataclass
class BraveConfig:
  """Specific configuration for the BraVe model."""

  # Broad view config
  include_video_in_broad_view: bool
  include_audio_in_broad_view: bool
  num_frames_narrow: int
  image_size_narrow: int
  step_broad: int

  # Narrow view config
  num_frames_broad: int
  image_size_broad: int
  step_narrow: int

  # Predictors and projectors use output_dims dimensions.
  output_dims: int
  tsm_resnet_width_multiplier: int

  # spectrogram config (when using audio)
  num_spectrogram_bins: int
  fft_step: int

  # Dataset
  dataset_shards: Sequence[str]
  input_video_sample_rate: int
  input_audio_sample_rate: int


def get_model(config: BraveConfig) -> embedding_model.MultimodalEmbeddingModel:
  """Construct a model implementing BraVe.

  Args:
    config: Configuration for BraVe.

  Returns:
    A `MultimodalEmbeddingModel` to train BraVe.
  """

  init_fn, parameterized_fns = _build_parameterized_fns(config)
  loss_fn = _build_loss_fn(config, parameterized_fns)
  forward_fns = {
      'broad_video': parameterized_fns.broad_video_embedding,
      'broad_audio': parameterized_fns.broad_audio_embedding,
      'narrow_video': parameterized_fns.narrow_video_embedding,
  }

  return embedding_model.MultimodalEmbeddingModel(
      init_fn=init_fn,
      forward_fns=forward_fns,
      loss_fn=loss_fn,
      evaluate_fn=_build_eval_fn(forward_fns),
      train_dataset_builder_fn=_train_dataset_builder(config),
  )


def get_empty_minibatch(config: BraveConfig) -> datasets.MiniBatch:
  """Get a zero-initialized minibatch for initialization and testing."""
  narrow_video = np.zeros([
      1, config.num_frames_narrow, config.image_size_narrow,
      config.image_size_narrow, 3
  ])

  broad_audio = None
  broad_video = None

  if config.include_video_in_broad_view:
    broad_video = np.zeros([
        1, config.num_frames_broad, config.image_size_broad,
        config.image_size_broad, 3
    ])

  if config.include_audio_in_broad_view:
    # Computing the actual size of this tensor is surprisingly difficult.
    # But in practice it doesn't matter, the parameter block will be the same
    # in this case, one simply has to re-jit.
    broad_audio = np.zeros([1, FAKE_AUDIO_LENGTH, config.num_spectrogram_bins])

  return datasets.MiniBatch(
      views={
          'broad':
              datasets.View(video=broad_video, audio=broad_audio, labels=None),
          'narrow':
              datasets.View(video=narrow_video, audio=None, labels=None),
      })


def _build_parameterized_fns(
    config: BraveConfig) -> Tuple[embedding_model.InitFn, ParameterizedFns]:
  """Initialize Brave trainable embedding functions and predictors.

  Args:
    config: Configuration for the brave model.

  Returns:
    All parameterized trainable functions used by the BraVe model.
  """
  output_dims = config.output_dims

  def broad_video_embedding(view, is_training):
    net = modules.VideoEmbedding(
        width_multiplier=config.tsm_resnet_width_multiplier,
        name='broad_video_embedding')
    return net(view, is_training)

  def broad_audio_embedding(view, is_training):
    net = modules.AudioEmbedding(name='broad_audio_embedding')
    return net(view, is_training)

  def narrow_video_embedding(view, is_training):
    net = modules.VideoEmbedding(
        width_multiplier=config.tsm_resnet_width_multiplier,
        name='narrow_video_embedding')
    return net(view, is_training)

  def b_video_to_n_video(f_b_1, is_training):
    net = modules.ProjectAndPredict(output_dims, name='b_video_to_n_video')
    return net(f_b_1, is_training)

  def n_video_to_b_video(f_n, is_training):
    net = modules.ProjectAndPredict(output_dims, name='n_video_to_b_video')
    return net(f_n, is_training)

  def b_audio_to_n_video(f_b_2, is_training):
    net = modules.ProjectAndPredict(output_dims, name='b_audio_to_n_video')
    return net(f_b_2, is_training)

  def n_video_to_b_audio(f_n, is_training):
    net = modules.ProjectAndPredict(output_dims, name='n_video_to_b_audio')
    return net(f_n, is_training)

  def init():
    batch = get_empty_minibatch(config)
    broad, narrow = batch.views['broad'], batch.views['narrow']
    f_n = narrow_video_embedding(narrow, is_training=True)

    if config.include_video_in_broad_view:
      f_b_1 = broad_video_embedding(broad, is_training=True)
      b_video_to_n_video(f_b_1, is_training=True)
      n_video_to_b_video(f_n, is_training=True)

    if config.include_audio_in_broad_view:
      f_b_2 = broad_audio_embedding(broad, is_training=True)
      b_audio_to_n_video(f_b_2, is_training=True)
      n_video_to_b_audio(f_n, is_training=True)

  return hk.transform_with_state(init).init, ParameterizedFns(
      broad_video_embedding=hk.transform_with_state(
          broad_video_embedding).apply,
      broad_audio_embedding=hk.transform_with_state(
          broad_audio_embedding).apply,
      narrow_video_embedding=hk.transform_with_state(
          narrow_video_embedding).apply,
      b_video_to_n_video=hk.transform_with_state(b_video_to_n_video).apply,
      n_video_to_b_video=hk.transform_with_state(n_video_to_b_video).apply,
      b_audio_to_n_video=hk.transform_with_state(b_audio_to_n_video).apply,
      n_video_to_b_audio=hk.transform_with_state(n_video_to_b_audio).apply,
  )


def _build_loss_fn(
    config: BraveConfig,
    paremeterized_fns: ParameterizedFns) -> embedding_model.LossFn:
  """Construct the loss function for BraVe.

  Takes as input the predictors across views, the predictors take as input
  a view and output the predicted value of a predictor computed from another
  view.

  Args:
    config: The config for BraVe.
    paremeterized_fns: The cross-view predictor functions.

  Returns:
    A function for computing the loss between the predictors with respect to
    the data contained within a minibatch.
  """

  broad_video_embedding = paremeterized_fns.broad_video_embedding
  broad_audio_embedding = paremeterized_fns.broad_audio_embedding
  narrow_video_embedding = paremeterized_fns.narrow_video_embedding

  b_video_to_n_video = paremeterized_fns.b_video_to_n_video
  n_video_to_b_video = paremeterized_fns.n_video_to_b_video
  b_audio_to_n_video = paremeterized_fns.b_audio_to_n_video
  n_video_to_b_audio = paremeterized_fns.n_video_to_b_audio

  def loss_fn(
      params: hk.Params,
      state: hk.State,
      rng: chex.PRNGKey,
      batch: datasets.MiniBatch,
  ) -> Tuple[chex.Array, Tuple[hk.State, embedding_model.Scalars]]:

    metrics = {}
    losses = []
    broad, narrow = batch.views['broad'], batch.views['narrow']
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 7)

    f_n, state = narrow_video_embedding(params, state, k1, narrow, True)

    if config.include_video_in_broad_view:
      f_b_1, state = broad_video_embedding(params, state, k2, broad, True)

      (z_b_1, h_b_1), state = b_video_to_n_video(params, state, k4, f_b_1, True)
      (z_n_1, h_n_1), state = n_video_to_b_video(params, state, k5, f_n, True)
      chex.assert_rank([z_b_1, h_b_1, z_n_1, h_n_1], 2)

      loss_b_to_n_1 = _regression_loss(h_b_1, jax.lax.stop_gradient(z_n_1))
      loss_n_to_b_1 = _regression_loss(h_n_1, jax.lax.stop_gradient(z_b_1))
      chex.assert_rank([loss_b_to_n_1, loss_b_to_n_1], 0)

      metrics['loss_b_to_n_1'] = loss_b_to_n_1
      metrics['loss_n_to_b_1'] = loss_n_to_b_1
      losses.extend([loss_b_to_n_1, loss_n_to_b_1])

    if config.include_audio_in_broad_view:
      f_b_2, state = broad_audio_embedding(params, state, k3, broad, True)

      (z_b_2, h_b_2), state = b_audio_to_n_video(params, state, k6, f_b_2, True)
      (z_n_2, h_n_2), state = n_video_to_b_audio(params, state, k7, f_n, True)
      chex.assert_rank([z_b_2, h_b_2, z_n_2, h_n_2], 2)

      loss_b_to_n_2 = _regression_loss(h_b_2, jax.lax.stop_gradient(z_n_2))
      loss_n_to_b_2 = _regression_loss(h_n_2, jax.lax.stop_gradient(z_b_2))
      chex.assert_rank([loss_b_to_n_2, loss_b_to_n_2], 0)

      metrics['loss_b_to_n_2'] = loss_b_to_n_2
      metrics['loss_n_to_b_2'] = loss_n_to_b_2
      losses.extend([loss_b_to_n_2, loss_n_to_b_2])

    loss = jnp.stack(losses).mean()
    chex.assert_rank(loss, 0)
    metrics['loss'] = loss

    return loss, (state, metrics)

  return loss_fn


def _regression_loss(x: chex.Array,
                     y: chex.Array,
                     epsilon: float = 1e-5) -> chex.Array:
  """Cosine-similarity based loss."""
  batched_norm_fn = jnp.vectorize(_safe_norm, signature='(k)->()', excluded={1})
  normed_x = x / jnp.expand_dims(batched_norm_fn(x, epsilon), axis=-1)
  normed_y = y / jnp.expand_dims(batched_norm_fn(y, epsilon), axis=-1)
  return jnp.mean(0.5 * jnp.sum((normed_x - normed_y)**2, axis=-1))


def _safe_norm(x: chex.Array, min_norm: float) -> chex.Array:
  """Compute normalization, with correct gradients."""
  norm = jnp.linalg.norm(x)
  x = jnp.where(norm <= min_norm, jnp.ones_like(x), x)
  return jnp.where(norm <= min_norm, min_norm, jnp.linalg.norm(x))


def _build_eval_fn(
    forward_fns: Dict[str, embedding_model.ForwardFn]
) -> embedding_model.EvaluateFn:
  """Construct a function to use for evaluating BraVe."""

  del forward_fns

  def eval_fn(global_step: int, mode: str, params: hk.Params,
              state: hk.State) -> Dict[str, chex.Array]:

    del mode
    del params
    del state

    # No online evaluation enabled in this release.
    return {'multiple_of_saving_period': global_step // (50 * 1000)}

  return eval_fn


def _train_dataset_builder(
    config: BraveConfig) -> embedding_model.DatasetBuilderFn:
  """Construct the train dataset for BraVe."""

  def build_dataset():
    return _train_dataset(config, config.dataset_shards, shuffle=True)

  return build_dataset


def _train_dataset(config: BraveConfig,
                   shards: Sequence[str],
                   *,
                   shuffle: bool = False) -> tf.data.Dataset:
  """Construct the train dataset for BraVe."""

  features = [media_sequences.FeatureKind.VIDEO]
  if config.include_audio_in_broad_view:
    features.append(media_sequences.FeatureKind.AUDIO)

  ds = datasets.multi_view_dataset(
      shards=shards,
      features=features,
      view_sampler=functools.partial(_brave_random_view_sampler, config),
      view_decoder=functools.partial(_brave_view_decoder, config),
      shuffle=shuffle)

  ds = ds.map(
      _transform_views,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False)

  return ds


def _transform_views(batch: datasets.MiniBatch) -> datasets.MiniBatch:
  """Apply data augmentations to the views in the batch."""

  narrow = batch.views['narrow']
  narrow_video_width = narrow.video.shape[-2]

  narrow = augmentations.normalize_video(narrow)
  narrow = augmentations.random_gaussian_blur_video(
      narrow, kernel_size=narrow_video_width // 10, sigma_range=(0.1, 2.0))
  narrow = augmentations.random_horizontal_flip_video(narrow)
  narrow = augmentations.random_color_augment_video(
      narrow, prob_color_augment=0.8, prob_color_drop=0.2)

  broad = batch.views['broad']
  if broad.video is not None:
    broad_video_width = broad.video.shape[-2]
    broad = augmentations.normalize_video(broad)
    broad = augmentations.random_gaussian_blur_video(
        broad, kernel_size=broad_video_width // 10, sigma_range=(0.1, 2.0))
    broad = augmentations.random_horizontal_flip_video(broad)
    broad = augmentations.random_color_augment_video(
        broad, prob_color_augment=0.8, prob_color_drop=0.2)

  result = copy.copy(batch)
  result.views = dict(broad=broad, narrow=narrow)
  return result


def _brave_random_view_sampler(
    config: BraveConfig, sequence: media_sequences.EncodedSequence
) -> Dict[str, media_sequences.EncodedSequence]:
  """Sample the data for BraVe."""

  # Extend the sequence so that there are enough frames
  # for the broad view.
  min_frames_required = (config.num_frames_broad - 1) * config.step_broad + 1
  sequence = media_sequences.extend_sequence(sequence, min_frames_required)

  num_audio_samples = _num_audio_samples(
      config.num_frames_broad * config.step_broad,
      config.input_video_sample_rate, config.input_audio_sample_rate)

  broad, broad_indices = time_sampling.random_sample_sequence_using_video(
      num_video_frames=config.num_frames_broad,
      video_frame_step=config.step_broad,
      sequence=sequence,
      override_num_audio_samples=num_audio_samples)

  narrow, _ = time_sampling.random_sample_sequence_using_video(
      num_video_frames=config.num_frames_narrow,
      video_frame_step=config.step_narrow,
      sequence=sequence,
      sample_start_index=broad_indices.start_index,
      sample_end_index=broad_indices.end_index)

  return {
      'broad': broad,
      'narrow': narrow,
  }


def _brave_view_decoder(
    config: BraveConfig, sequences: Dict[str, media_sequences.EncodedSequence]
) -> Dict[str, datasets.View]:
  """Decode sequences to return the views for BraVe."""
  result = {}

  for view_name, sequence in sequences.items():
    result[view_name] = datasets.View(
        video=_get_video_for_view(config, view_name, sequence),
        audio=_get_audio_for_view(config, view_name, sequence),
        labels=None)

  return result


def _get_video_for_view(
    config: BraveConfig, view_name: str,
    sequence: media_sequences.EncodedSequence) -> Optional[tf.Tensor]:
  """Sample and decode video."""

  if view_name == 'broad':
    if config.include_video_in_broad_view:
      return _sample_video(config, sequence, config.image_size_broad)
    else:
      return None

  return _sample_video(config, sequence, config.image_size_narrow)


def _get_audio_for_view(
    config: BraveConfig, view_name: str,
    sequence: media_sequences.EncodedSequence) -> Optional[tf.Tensor]:
  """Get the audio field for the view, if needed."""

  if config.include_audio_in_broad_view and view_name == 'broad':
    return spectrograms.pcm_to_log_mel_spectrogram(
        sequence.audio,
        input_sample_rate=config.input_audio_sample_rate,
        num_spectrogram_bins=config.num_spectrogram_bins,
        fft_step=config.fft_step)

  return None


def _sample_video(config: BraveConfig,
                  sequence: media_sequences.EncodedSequence,
                  image_size: int) -> tf.Tensor:
  """Randomly crop and decode videos to a given square image size."""

  del config

  # Extract shape only reads the image header.
  image_shape = tf.image.extract_jpeg_shape(sequence.jpeg_encoded_images[0])

  crop_window = video_sampling.random_sample_crop_window(
      image_shape,
      min_area=0.3,
      max_area=1.0,
      min_aspect_ratio=0.5,
      max_aspect_ratio=2.0)

  return video_sampling.decode_crop_resize_images(
      sequence.jpeg_encoded_images,
      crop_window,
      image_size=(image_size, image_size))


def _num_audio_samples(num_video_frames: int, video_sample_rate: int,
                       audio_sample_rate: int) -> int:
  return int((num_video_frames / video_sample_rate) * audio_sample_rate)
