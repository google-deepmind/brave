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

"""Implement trainable Haiku modules for Brave."""

from typing import Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from brave.datasets import datasets
from brave.models.brave import resnet
from brave.models.brave import tsm_resnet

DEFAULT_EMBEDDING_DIMS = 2048
DEFAULT_AUDIO_BACKBONE_DEPTH = 50


class ProjectAndPredict(hk.Module):

  def __init__(self, output_dims: int, name: Optional[str] = None):
    super(ProjectAndPredict, self).__init__(name=name)
    self._output_dims = output_dims

  def __call__(self, feats: chex.Array, is_training: bool) -> chex.Array:
    z = Projector(self._output_dims)(feats, is_training)
    h = Predictor(self._output_dims, name='predictor')(z, is_training)
    return z, h


class AudioEmbedding(hk.Module):
  """Compute an audio embedding for spectrogram audio."""

  def __init__(self, name: Optional[str] = None):
    super(AudioEmbedding, self).__init__(name=name)

  def __call__(self, view: datasets.View, is_training: bool) -> chex.Array:
    assert view.audio is not None

    net = resnet.ResNetV2(
        depth=DEFAULT_AUDIO_BACKBONE_DEPTH,
        normalize_fn=_default_normalize_fn,
        num_classes=None)

    audio = jnp.expand_dims(view.audio, axis=-1)
    result = net(audio, is_training=is_training)
    chex.assert_shape(result, (None, DEFAULT_EMBEDDING_DIMS))

    return result


class VideoEmbedding(hk.Module):
  """Given a view, compute an embedding."""

  def __init__(self, width_multiplier: int, name: Optional[str] = None):
    super(VideoEmbedding, self).__init__(name=name)
    self.width_multiplier = width_multiplier

  def __call__(self, view: datasets.View, is_training: bool) -> chex.Array:
    assert view.video is not None

    chex.assert_shape(view.video, (None, None, None, None, 3))  # B, T, H, W, C
    net = tsm_resnet.TSMResNetV2(
        normalize_fn=_default_normalize_fn, width_mult=self.width_multiplier)
    feats = net(view.video, is_training=is_training)

    expected_output_dims = self.width_multiplier * DEFAULT_EMBEDDING_DIMS
    chex.assert_shape(feats, (None, expected_output_dims))

    return feats


class Projector(hk.Module):
  """Project backbone features into representation space."""

  def __init__(self, output_dims: int, name: Optional[str] = None):
    super(Projector, self).__init__(name=name)
    self.output_dims = output_dims

  def __call__(self, x: chex.Array, is_training: bool) -> chex.Array:
    x = hk.Linear(4096)(x)
    x = _default_normalize_fn(x, is_training)
    x = jax.nn.relu(x)
    x = hk.Linear(self.output_dims)(x)
    x = _default_normalize_fn(x, is_training)

    return x


class Predictor(hk.Module):
  """Take projected vector and predict the projected space of another view."""

  def __init__(self, output_dims: int, name: Optional[str] = None):
    super(Predictor, self).__init__(name=name)
    self.output_dims = output_dims

  def __call__(self, x: chex.Array, is_training: bool) -> chex.Array:
    """Project a projected z to predict another projected z."""

    x = hk.Linear(4096)(x)
    x = _default_normalize_fn(x, is_training)
    x = jax.nn.relu(x)

    return hk.Linear(self.output_dims)(x)


def _default_normalize_fn(x: chex.Array, is_training: bool):
  return hk.BatchNorm(
      create_scale=True,
      create_offset=True,
      decay_rate=0.9,
      cross_replica_axis='i',
  )(x, is_training)
