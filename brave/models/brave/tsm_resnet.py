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

"""Temporal Shift Module w/ ResNet-50 and ResNet-101.

Based on:
  TSM: Temporal Shift Module for Efficient Video Understanding
  Ji Lin, Chuang Gan, Song Han
  https://arxiv.org/pdf/1811.08383.pdf.
"""

from typing import Callable, Optional, Tuple

from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp

NormalizeFn = Callable[..., chex.Array]


class TSMResNetBlock(hk.Module):
  """A ResNet subblock with Temporal Channel Shifting.

  Combines a typical ResNetV2 block implementation
  (see https://arxiv.org/abs/1512.03385) with a pre-convolution Temporal
  Shift Module (see https://arxiv.org/pdf/1811.08383.pdf) in the residual.
  """

  def __init__(self,
               output_channels: int,
               stride: int,
               use_projection: bool,
               tsm_mode: str,
               normalize_fn: Optional[NormalizeFn] = None,
               channel_shift_fraction: float = 0.125,
               num_frames: int = 8,
               name: str = 'TSMResNetBlock'):
    """Initializes the TSMResNetBlock module.

    Args:
      output_channels: Number of output channels.
      stride: Stride used in convolutions.
      use_projection: Whether to use a projection for the shortcut.
      tsm_mode: Mode for TSM ('gpu' or 'tpu' or 'deflated_0.x').
      normalize_fn: Function used for normalization.
      channel_shift_fraction: The fraction of temporally shifted channels. If
        `channel_shift_fraction` is 0, the block is the same as a normal ResNet
        block.
      num_frames: Size of frame dimension in a single batch example
      name: The name of the module.
    """
    super().__init__(name=name)
    self._output_channels = output_channels
    self._bottleneck_channels = output_channels // 4
    self._stride = stride
    self._use_projection = use_projection
    self._normalize_fn = normalize_fn
    self._tsm_mode = tsm_mode
    self._channel_shift_fraction = channel_shift_fraction
    self._num_frames = num_frames

  def __call__(self,
               inputs: chex.Array,
               is_training: bool = True) -> jnp.ndarray:
    """Connects the ResNetBlock module into the graph.

    Args:
      inputs: A 4-D float array of shape `[B, H, W, C]`.
      is_training: Whether to use training mode.

    Returns:
      A 4-D float array of shape
      `[B * num_frames, new_h, new_w, output_channels]`.
    """
    # ResNet V2 uses pre-activation, where the batch norm and relu are before
    # convolutions, rather than after as in ResNet V1.
    preact = inputs
    if self._normalize_fn is not None:
      preact = self._normalize_fn(preact, is_training=is_training)
    preact = jax.nn.relu(preact)

    if self._use_projection:
      shortcut = hk.Conv2D(
          output_channels=self._output_channels,
          kernel_shape=1,
          stride=self._stride,
          with_bias=False,
          padding='SAME',
          name='shortcut_conv')(
              preact)
    else:
      shortcut = inputs

    # Eventually applies Temporal Shift Module.
    if self._channel_shift_fraction != 0:
      preact = apply_temporal_shift(
          preact,
          tsm_mode=self._tsm_mode,
          num_frames=self._num_frames,
          channel_shift_fraction=self._channel_shift_fraction)

    # First convolution.
    residual = hk.Conv2D(
        self._bottleneck_channels,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding='SAME',
        name='conv_0')(
            preact)

    # Second convolution.
    if self._normalize_fn is not None:
      residual = self._normalize_fn(residual, is_training=is_training)
    residual = jax.nn.relu(residual)
    residual = hk.Conv2D(
        output_channels=self._bottleneck_channels,
        kernel_shape=3,
        stride=self._stride,
        with_bias=False,
        padding='SAME',
        name='conv_1')(
            residual)

    # Third convolution.
    if self._normalize_fn is not None:
      residual = self._normalize_fn(residual, is_training=is_training)
    residual = jax.nn.relu(residual)
    residual = hk.Conv2D(
        output_channels=self._output_channels,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding='SAME',
        name='conv_2')(
            residual)

    # NOTE: we do not use block multiplier.
    output = shortcut + residual
    return output


class TSMResNetUnit(hk.Module):
  """Block group for TSM ResNet."""

  def __init__(self,
               output_channels: int,
               num_blocks: int,
               stride: int,
               tsm_mode: str,
               num_frames: int,
               normalize_fn: Optional[NormalizeFn] = None,
               channel_shift_fraction: float = 0.125,
               name: str = 'tsm_resnet_unit'):
    """Creates a TSMResNet Unit.

    Args:
      output_channels: Number of output channels.
      num_blocks: Number of ResNet blocks in the unit.
      stride: Stride of the unit.
      tsm_mode: Which temporal shift module to use.
      num_frames: Size of frame dimension in a single batch example.
      normalize_fn: Function used for normalization.
      channel_shift_fraction: The fraction of temporally shifted channels. If
        `channel_shift_fraction` is 0, the block is the same as a normal ResNet
        block.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._output_channels = output_channels
    self._num_blocks = num_blocks
    self._normalize_fn = normalize_fn
    self._stride = stride
    self._tsm_mode = tsm_mode
    self._channel_shift_fraction = channel_shift_fraction
    self._num_frames = num_frames

  def __call__(self, inputs: chex.Array, is_training: bool) -> jnp.ndarray:
    """Connects the module to inputs.

    Args:
      inputs: A 4-D float array of shape `[B * num_frames, H, W, C]`.
      is_training: Whether to use training mode.

    Returns:
      A 4-D float array of shape
      `[B * num_frames, H // stride, W // stride, output_channels]`.
    """
    net = inputs
    for idx_block in range(self._num_blocks):
      net = TSMResNetBlock(
          self._output_channels,
          stride=self._stride if idx_block == 0 else 1,
          use_projection=idx_block == 0,
          normalize_fn=self._normalize_fn,
          tsm_mode=self._tsm_mode,
          channel_shift_fraction=self._channel_shift_fraction,
          num_frames=self._num_frames,
          name=f'block_{idx_block}')(
              net, is_training=is_training)
    return net


class TSMResNetV2(hk.Module):
  """TSM based on ResNet V2 as described in https://arxiv.org/abs/1603.05027."""

  # Endpoints of the model in order.
  VALID_ENDPOINTS = (
      'tsm_resnet_stem',
      'tsm_resnet_unit_0',
      'tsm_resnet_unit_1',
      'tsm_resnet_unit_2',
      'tsm_resnet_unit_3',
      'last_conv',
      'Embeddings',
  )

  def __init__(self,
               normalize_fn: Optional[NormalizeFn] = None,
               depth: int = 50,
               num_frames: int = 16,
               channel_shift_fraction: float = 0.125,
               width_mult: int = 1,
               name: str = 'TSMResNetV2'):
    """Constructs a ResNet model.

    Args:
      normalize_fn: Function used for normalization.
      depth: Depth of the desired ResNet.
      num_frames: Number of frames (used in TPU mode).
      channel_shift_fraction: Fraction of channels that are temporally shifted,
        if `channel_shift_fraction` is 0, a regular ResNet is returned.
      width_mult: Whether or not to use a width multiplier.
      name: The name of the module.

    Raises:
      ValueError: If `channel_shift_fraction` or `depth` has invalid value.
    """
    super().__init__(name=name)

    if not 0. <= channel_shift_fraction <= 1.0:
      raise ValueError(f'channel_shift_fraction ({channel_shift_fraction})'
                       ' has to be in [0, 1].')

    self._num_frames = num_frames

    self._channels = (256, 512, 1024, 2048)
    self._strides = (1, 2, 2, 2)

    num_blocks = {
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
        200: (3, 24, 36, 3),
    }
    if depth not in num_blocks:
      raise ValueError(
          f'`depth` should be in {list(num_blocks.keys())} ({depth} given).')
    self._num_blocks = num_blocks[depth]

    self._width_mult = width_mult
    self._channel_shift_fraction = channel_shift_fraction
    self._normalize_fn = normalize_fn

  def __call__(self,
               inputs: chex.Array,
               is_training: bool = True,
               final_endpoint: str = 'Embeddings',
               is_deflated: bool = False,
               alpha_deflation: float = 0.3) -> jnp.ndarray:
    """Connects the TSM ResNetV2 module into the graph.

    Args:
      inputs: The input may be in one of two shapes; if the shape is `[B, T, H,
        W, C]`, this module assumes the backend is a GPU (setting
        `tsm_mode='gpu'`) and `T` is treated the time dimension, with `B` being
        the batch dimension. This mode cannot be used when `is_deflated` is
        `true`. In this mode, the num_frames parameter passed to the constructor
        is ignored. If the shape is `[B, H, W, C]`, then the batch dimension is
        assumed to be of the form [B*T, H, W, C], where `T` is the number of
        frames in each video. This value may be set by passing `num_frames=n` to
        the constructor. The default value is `n=16` (beware this default is not
        the same as the default for the `TSMResNetBlock`, which has a default of
        8 frames). In this case, the module assumes it is being run on a TPU,
        and emits instructions that are more efficient for for that case,
        using`tsm_mode`='tpu'` for the downstream blocks.
      is_training: Whether to use training mode.
      final_endpoint: Up to which endpoint to run / return.
      is_deflated: Whether or not to use the deflated version of the network.
      alpha_deflation: Deflation parameter to use for dealing with the padding
        effect.

    Returns:
      Network output at location `final_endpoint`. A float array which shape
      depends on `final_endpoint`.

    Raises:
      ValueError: If `final_endpoint` is not recognized.
    """

    # Prepare inputs for TSM.
    if is_deflated:
      if len(inputs.shape) != 4:
        raise ValueError(
            'In deflated mode inputs should be given as [B, H, W, 3]')
      logging.warning(
          'Deflation is an experimental feature and the API might change.')
      tsm_mode = f'deflated_{alpha_deflation}'
      num_frames = 1
    else:
      inputs, tsm_mode, num_frames = prepare_inputs(inputs)
      num_frames = num_frames or self._num_frames

    self._final_endpoint = final_endpoint
    if self._final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError(f'Unknown final endpoint {self._final_endpoint}')

    # Stem convolution.
    end_point = 'tsm_resnet_stem'
    net = hk.Conv2D(
        output_channels=64 * self._width_mult,
        kernel_shape=7,
        stride=2,
        with_bias=False,
        name=end_point,
        padding='SAME')(
            inputs)
    net = hk.MaxPool(
        window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')(
            net)
    if self._final_endpoint == end_point:
      net = prepare_outputs(net, tsm_mode, num_frames, reduce_mean=False)
      return net

    # Residual block.
    for unit_id, (channels, num_blocks, stride) in enumerate(
        zip(self._channels, self._num_blocks, self._strides)):
      end_point = f'tsm_resnet_unit_{unit_id}'
      net = TSMResNetUnit(
          output_channels=channels * self._width_mult,
          num_blocks=num_blocks,
          stride=stride,
          normalize_fn=self._normalize_fn,
          channel_shift_fraction=self._channel_shift_fraction,
          num_frames=num_frames,
          tsm_mode=tsm_mode,
          name=end_point)(
              net, is_training=is_training)
      if self._final_endpoint == end_point:
        net = prepare_outputs(net, tsm_mode, num_frames, reduce_mean=False)
        return net

    if self._normalize_fn is not None:
      net = self._normalize_fn(net, is_training=is_training)
    net = jax.nn.relu(net)

    end_point = 'last_conv'
    if self._final_endpoint == end_point:
      net = prepare_outputs(net, tsm_mode, num_frames, reduce_mean=False)
      return net
    net = jnp.mean(net, axis=(1, 2))
    # Prepare embedding outputs for TSM (temporal average of features).
    net = prepare_outputs(net, tsm_mode, num_frames, reduce_mean=True)
    assert self._final_endpoint == 'Embeddings'
    return net


def prepare_inputs(inputs: chex.Array) -> Tuple[jnp.ndarray, str, int]:
  """Deduces input mode for TSM."""
  # Deduce if we run on TPU based on input shape.
  if len(inputs.shape) == 5:
    # Input is given in the standard [B, T, H, W, 3] format.
    tsm_mode = 'gpu'
    num_frames = inputs.shape[1]
    inputs = jnp.reshape(inputs, [-1] + list(inputs.shape[2:]))
  else:
    # Input is given in the [T * B, H, W, 3] format.
    tsm_mode = 'tpu'
    num_frames = None
  return inputs, tsm_mode, num_frames


def prepare_outputs(outputs: chex.Array,
                    tsm_mode: str,
                    num_frames: int,
                    reduce_mean: bool = True) -> jnp.ndarray:
  """Processes output of TSM to undo the merging of batch and time."""
  # Get the shape without the batch/time dimension (for TSM batch and time are
  # merged in the first dimension).
  shape_no_bt = list(outputs.shape[1:])
  if tsm_mode == 'tpu':
    # Outputs are of the shape [num_frames * B, ..., n_channels]
    outputs = jnp.reshape(outputs, [num_frames, -1] + shape_no_bt)
    if reduce_mean:
      # We average over time and space.
      outputs = jnp.mean(
          outputs, axis=[0] + list(range(2,
                                         len(shape_no_bt) + 1)))
    else:
      outputs = jnp.transpose(
          outputs, axes=[1, 0] + list(range(2,
                                            len(shape_no_bt) + 2)))
  elif tsm_mode == 'gpu':
    # Outputs are of the shape [B * num_frames, ..., n_channels]
    outputs = jnp.reshape(outputs, [-1, num_frames] + shape_no_bt)
    if reduce_mean:
      outputs = jnp.mean(
          outputs, axis=[1] + list(range(2,
                                         len(shape_no_bt) + 1)))
  elif tsm_mode.startswith('deflated'):
    # In deflated mode, outputs are already in the right format.
    pass
  else:
    raise ValueError('`tsm_mode` should be \'tpu\' or \'gpu\' or '
                     f'\'deflated_0.x\' ({tsm_mode} given)')
  return outputs


def apply_temporal_shift(x: chex.Array,
                         tsm_mode: str,
                         num_frames: int,
                         channel_shift_fraction: float = 0.125) -> jnp.ndarray:
  """Performs a temporal shift: https://arxiv.org/abs/1811.08383 with mode."""
  if tsm_mode == 'tpu':
    outputs = temporal_shift_tpu(x, num_frames, channel_shift_fraction)
  elif tsm_mode == 'gpu':
    outputs = temporal_shift_gpu(x, num_frames, channel_shift_fraction)
  elif tsm_mode.startswith('deflated'):
    alpha = float(tsm_mode.split('_')[1])
    outputs = temporal_shift_image_mode(x, channel_shift_fraction, alpha)
  else:
    raise ValueError('`tsm_mode` should be \'tpu\' or \'gpu\' or '
                     f'\'deflated_0.x\' ({tsm_mode} given)')
  return outputs


def temporal_shift_image_mode(x, channel_shift_fraction=0.125, alpha=0.3):
  """Temporal shift applied on single image (to emulate a fixed video)."""
  # B, H, W, C = batch_size, im_height, im_width, channels
  # Input is (B, H, W, C)
  orig_shp = tuple(x.shape)
  n_channels = orig_shp[-1]
  n_shift = int(n_channels * channel_shift_fraction)
  # Alpha emulates the effect of the padding when using a single frame
  shifted_backward = alpha * x[:, :, :, -n_shift:]
  shifted_forward = alpha * x[:, :, :, :n_shift]
  no_shift = x[:, :, :, n_shift:-n_shift]
  shifted_x = jnp.concatenate([shifted_backward, no_shift, shifted_forward],
                              axis=3)
  return shifted_x


def temporal_shift_gpu(x: chex.Array,
                       num_frames: int,
                       channel_shift_fraction: float = 0.125) -> jnp.ndarray:
  """Performs a temporal shift: https://arxiv.org/abs/1811.08383."""
  # B, T, H, W, C = batch_size, num_frames, im_height, im_width, channels
  # Input is (B * T, H, W, C)
  orig_shp = tuple(x.shape)
  reshaped_x = jnp.reshape(x, (-1, num_frames) + orig_shp[1:])
  n_channels = orig_shp[-1]
  n_shift = int(n_channels * channel_shift_fraction)

  new_shp = tuple(reshaped_x.shape)

  # shifted_backward = reshaped_x[:, 1:, :, :, -n_shift:]
  shifted_backward = jax.lax.slice(
      reshaped_x, (0, 1, 0, 0, new_shp[4] - n_shift),
      (new_shp[0], new_shp[1], new_shp[2], new_shp[3], new_shp[4]))
  shifted_backward_padding = ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0))
  shifted_backward = jnp.pad(shifted_backward, shifted_backward_padding)

  # shifted_forward = reshaped_x[:, :-1, :, :, :n_shift]
  shifted_forward = jax.lax.slice(
      reshaped_x, (0, 0, 0, 0, 0),
      (new_shp[0], new_shp[1] - 1, new_shp[2], new_shp[3], n_shift))
  shifted_forward_padding = ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0))
  shifted_forward = jnp.pad(shifted_forward, shifted_forward_padding)

  no_shift = reshaped_x[:, :, :, :, n_shift:-n_shift]
  shifted_x = jnp.concatenate([shifted_backward, no_shift, shifted_forward],
                              axis=4)
  return jnp.reshape(shifted_x, (-1,) + orig_shp[1:])


def temporal_shift_tpu(x: chex.Array,
                       num_frames: int,
                       channel_shift_fraction: float = 0.125) -> jnp.ndarray:
  """Performs a temporal shift: https://arxiv.org/abs/1811.08383.

    TPU optimized version of TSM.
  Args:
    x: Input expected to be [T * B, H, W, C] (where the batch has been reshaped
      from a time major version of the input).
    num_frames: number of frames T per video.
    channel_shift_fraction: fraction of the channel to shift forward and
      backward.

  Returns:
      The temporal shifted version of x.
  """
  # B, T, H, W, C = batch_size, num_frames, im_height, im_width, channels
  # Input is (T * B, H, W, C)
  original_dtype = x.dtype
  original_shape = list(x.shape)

  batch_size = int(original_shape[0] / num_frames)
  n_channels = int(original_shape[-1])
  n_shift = int(n_channels * channel_shift_fraction)

  # Cast to bfloat16.
  x = x.astype(jnp.bfloat16)

  # For the following, assume that x has 3 channels [x1, x2, x3] and n_shift=1.
  # Shift backward, we first pad by zeros [x1, x2, x3, 0, 0].
  orig_shp = list(x.shape)

  shifted_backward_padding = ((0, batch_size, 0), (0, 0, 0), (0, 0, 0),
                              (0, n_channels - n_shift, 0))
  x_backward_padding = jax.lax.pad(
      x,
      padding_value=jnp.bfloat16(0.),
      padding_config=shifted_backward_padding)
  # The following shift gets to [x3^+1, 0, 0] (where +1 means from the future).
  shifted_backward = jax.lax.slice(x_backward_padding,
                                   (batch_size, 0, 0, n_channels - n_shift),
                                   (orig_shp[0] + batch_size, orig_shp[1],
                                    orig_shp[2], 2 * n_channels - n_shift))
  # Shift forward, we first pad by zeros [0, 0, x1, x2, x3].
  shifted_forward_padding = ((batch_size, 0, 0), (0, 0, 0), (0, 0, 0),
                             (n_channels - n_shift, 0, 0))
  x_forward_padding = jax.lax.pad(
      x, padding_value=jnp.bfloat16(0.), padding_config=shifted_forward_padding)
  # The following shift gets to [0, 0, x1^-1] (where -1 means from the past).
  shifted_forward = jax.lax.slice(
      x_forward_padding, (0, 0, 0, 0),
      (orig_shp[0], orig_shp[1], orig_shp[2], n_channels))
  # No shift is in the middle, this gets [0, x2, 0].
  mask_noshift = (jnp.reshape((jnp.arange(n_channels) >= n_shift) &
                              (jnp.arange(n_channels) < n_channels - n_shift),
                              (1, 1, 1, -1))).astype(jnp.bfloat16)
  no_shift = mask_noshift * x
  # By summing everything together, we end up with [x3^+1, x2, x1^-1].
  # Note: channels have been reordered but that doesn't matter for the model.
  shifted_x = shifted_backward + shifted_forward + no_shift

  return shifted_x.astype(original_dtype)
