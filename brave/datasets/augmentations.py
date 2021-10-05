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

"""Provide view augmentations for the dataset."""

import copy
import functools
from typing import Optional, Tuple

import tensorflow as tf

from brave.datasets import datasets

DEFAULT_RANDOM_CONVOLVE_MAX_K = 11


def normalize_video(view: datasets.View) -> datasets.View:
  """Return view with video normalized to range [0, 1]."""

  result = copy.copy(view)
  result.video = view.video * (1.0 / 255.0)
  return result


def random_color_augment_video(view: datasets.View, *,
                               prob_color_augment: float,
                               prob_color_drop: float) -> datasets.View:
  """Apply random color augmentations to the video in a view."""

  video = _color_default_augm(
      view.video,
      zero_centering_image=False,
      prob_color_augment=prob_color_augment,
      prob_color_drop=prob_color_drop)

  result = copy.copy(view)
  result.video = video
  return result


def random_gaussian_blur_video(
    view: datasets.View, *, kernel_size: int,
    sigma_range: Tuple[float, float]) -> datasets.View:
  """Apply a gaussian blur with a random sigma value in the range sigma_range.

  Args:
    view: The input view to augment.
    kernel_size: The kernel size of the blur kernel.
    sigma_range: A random value in this range is chosen as the sigma value for
      the gaussian blur.

  Returns:
    A new view where the video has a guassian gaussian blur applied.
  """

  sigma = tf.random.uniform((),
                            sigma_range[0],
                            sigma_range[1],
                            dtype=tf.float32)

  def blur(img):
    return _gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

  result = copy.copy(view)
  result.video = tf.map_fn(blur, view.video, fn_output_signature=tf.float32)
  return result


def random_horizontal_flip_video(view: datasets.View) -> datasets.View:
  """Randomly flip all frames within a video."""

  flip = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
  video = tf.cond(
      pred=tf.equal(flip, 1),
      true_fn=lambda: tf.image.flip_left_right(view.video),
      false_fn=lambda: view.video)

  result = copy.copy(view)
  result.video = video
  return result


def random_convolve_video(view: datasets.View,
                          *,
                          max_k=DEFAULT_RANDOM_CONVOLVE_MAX_K) -> datasets.View:
  """Apply a random convolution to the input view's video."""

  video = _random_convolve(view.video, max_k=max_k)
  result = copy.copy(view)
  result.video = video
  return result


def _gaussian_blur(image: tf.Tensor,
                   kernel_size: int,
                   sigma: float,
                   padding='SAME'):
  """Blurs the given image with separable convolution.

  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be
      size + 1.
    sigma: Sigma value for gaussian operator.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.

  Returns:
    A Tensor representing the blurred image.
  """
  radius = tf.cast(kernel_size // 2, tf.int32)
  kernel_size = radius * 2 + 1
  x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
  blur_filter = tf.exp(-tf.pow(x, 2.0) /
                       (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0)))
  blur_filter /= tf.reduce_sum(blur_filter)
  # One vertical and one horizontal filter.
  blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
  blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
  num_channels = tf.shape(image)[-1]
  blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
  blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
  expand_batch_dim = image.shape.ndims == 3
  if expand_batch_dim:
    # Tensorflow requires batched input to convolutions, which we can fake with
    # an extra dimension.
    image = tf.expand_dims(image, axis=0)
  blurred = tf.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
  blurred = tf.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
  if expand_batch_dim:
    blurred = tf.squeeze(blurred, axis=0)
  return blurred


def _color_default_augm(frames: tf.Tensor,
                        zero_centering_image: bool = False,
                        prob_color_augment: float = 0.8,
                        prob_color_drop: float = 0.0,
                        seed: Optional[int] = None):
  """Standard color augmentation for videos.

  Args:
    frames: A float32 tensor of shape [timesteps, input_h, input_w, channels].
    zero_centering_image: If `True`, results are in [-1, 1], if `False`, results
      are in [0, 1].
    prob_color_augment: Probability of applying color augmentation.
    prob_color_drop: Probability of droping the colors to gray scale.
    seed: A seed to use for the random sampling.

  Returns:
    A tensor of same shape as the input with color eventually altered.
  """

  def color_augment(video: tf.Tensor) -> tf.Tensor:
    """Do standard color augmentations."""
    # Note the same augmentation will be applied to all frames of the video.
    if zero_centering_image:
      video = 0.5 * (video + 1.0)
    video = tf.image.random_brightness(video, max_delta=32. / 255.)
    video = tf.image.random_saturation(video, lower=0.6, upper=1.4)
    video = tf.image.random_contrast(video, lower=0.6, upper=1.4)
    video = tf.image.random_hue(video, max_delta=0.2)
    video = tf.clip_by_value(video, 0.0, 1.0)
    if zero_centering_image:
      video = 2 * (video - 0.5)
    return video

  def color_drop(video: tf.Tensor) -> tf.Tensor:
    """Do color drop."""
    video = tf.image.rgb_to_grayscale(video)
    video = tf.tile(video, [1, 1, 1, 3])
    return video

  should_color_augment = tf.random.uniform([],
                                           minval=0,
                                           maxval=1,
                                           dtype=tf.float32,
                                           seed=seed)
  frames = tf.cond(
      pred=tf.less(should_color_augment, tf.cast(prob_color_augment,
                                                 tf.float32)),
      true_fn=lambda: color_augment(frames),
      false_fn=lambda: frames)

  should_color_drop = tf.random.uniform([],
                                        minval=0,
                                        maxval=1,
                                        dtype=tf.float32,
                                        seed=seed)
  frames = tf.cond(
      pred=tf.less(should_color_drop, tf.cast(prob_color_drop, tf.float32)),
      true_fn=lambda: color_drop(frames),
      false_fn=lambda: frames)

  return frames


def _random_convolve(x: tf.Tensor, max_k: int, init='he') -> tf.Tensor:
  """Applies a random convolution of random odd kernel size <= max_k."""

  if init == 'he':
    he_normal_init = tf.initializers.he_normal
    w_init = he_normal_init()
  else:
    raise NotImplementedError(f'Unknown init: {init} for RandConv.')

  _, _, _, ch = x.get_shape().as_list()

  # Prepare the switch case operation, depending on the dynamically sampled k.
  values_k = range(1, max_k + 1, 2)
  nb_values_k = len(values_k)
  random_conv_fns = {}

  def apply_conv2d_fn(x, k, ch, w_init):
    k_h, k_w, k_ic, k_oc = k, k, ch, ch
    w_shape = [k_h, k_w, k_ic, k_oc]
    strides = 1
    w = w_init(w_shape)
    return tf.nn.conv2d(x, w, strides, 'SAME', name='random_conv')

  for ind_k in range(nb_values_k):
    k = 2 * ind_k + 1
    apply_conv_k_fn = functools.partial(apply_conv2d_fn, x, k, ch, w_init)
    random_conv_fns[ind_k] = apply_conv_k_fn

  # Sample k uniformly in 1:max_k:2.
  ind_k = tf.cast(tf.floor(tf.random.uniform([], maxval=nb_values_k)), tf.int32)
  x = tf.switch_case(ind_k, random_conv_fns, name='sample_random_conv')

  return x
