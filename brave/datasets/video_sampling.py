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

"""A package to sample from videos."""

from typing import Tuple, Optional, Union

import tensorflow as tf

DEFAULT_MAX_RANDOM_SAMPLE_CROP_ATTEMPTS = 20
DEFAULT_PADDING_ON_FAIL = 16


def pad_and_center_crop_window(image_shape: tf.Tensor,
                               padding: int = 16) -> tf.Tensor:
  """Compute a crop window for a padded center crop of the given image shape.

  Args:
    image_shape: The shape of the jpeg [height, width, channels], or [height,
      width].
    padding: The padding between the input image and the resulting image. The
      padding represents the distance between the input image and the output
      image at each edge (so that the total number of pixels removed from the
      smallest edge is 2 X the padding value.

  Returns:
    A crop window [y, x, image_size, image_size],
    where image_size = min(height, width) - 2 * padding, and y and x are
    chosen so that the resutling crop falls in the center of the input image.
  """
  # Scrub the channels size, if it was provided.
  image_shape = image_shape[:2]

  min_image_side = tf.math.reduce_min(image_shape)
  image_height = image_shape[0]
  image_width = image_shape[1]

  # If the padding is larger than the image, no pixels will be returned.
  tf.debugging.assert_greater(min_image_side, 2 * padding)

  offset_y = tf.cast((image_height - min_image_side) / 2, dtype=tf.int32)
  offset_x = tf.cast((image_width - min_image_side) / 2, dtype=tf.int32)

  image_size = tf.cast(min_image_side - 2 * padding, tf.int32)
  return tf.stack(
      [offset_y + padding, offset_x + padding, image_size, image_size])


def random_sample_crop_window(image_shape: tf.Tensor,
                              min_area: float,
                              max_area: float,
                              min_aspect_ratio: float,
                              max_aspect_ratio: float,
                              padding_on_fail: int = DEFAULT_PADDING_ON_FAIL,
                              seed: Optional[int] = None) -> tf.Tensor:
  """Randomly sample a crop window, given an image size and config.

  It may be that the random sampler is unable to satisfy the constraints given
  (within an acceptable number of iterations). In this case, the sampler
  falls back to returning the result of `pad_and_center_crop_window`, with the
  default padding set.

  Args:
    image_shape: A tensor containing [image_height, image_width, channels].
    min_area: A float with the minimum area used per crop.
    max_area: A float with the maximum area used per crop.
    min_aspect_ratio: A float with the minimum aspect ratio distorsion for the
      crops.
    max_aspect_ratio: A float with the maximum aspect ratio distorsion for the
      crops.
    padding_on_fail: The padding to use if the sampler fails to return a valid
      sample.
    seed: The seed to pass to the random sampler.

  Returns:
    A bounding box tensor [min y, min x, height, width] in image coordinates.
  """

  crop_window = _sample_crop_window(
      image_shape=image_shape,
      min_object_covered=min_area,
      aspect_ratio_range=(min_aspect_ratio, max_aspect_ratio),
      area_range=(min_area, max_area),
      max_attempts=DEFAULT_MAX_RANDOM_SAMPLE_CROP_ATTEMPTS,
      seed=seed)

  # If the random crop failed, fall back to padded center crop.
  return tf.cond(
      tf.reduce_all(tf.equal(image_shape[:2], crop_window[2:])),
      lambda: pad_and_center_crop_window(image_shape, padding=padding_on_fail),
      lambda: tf.identity(crop_window))


def decode_crop_resize_images(
    jpeg_encoded_images: tf.Tensor, crop_window: tf.Tensor,
    image_size: Union[tf.Tensor, Tuple[int, int]]) -> tf.Tensor:
  """Given a crop window, decode the input tensors.

  Args:
    jpeg_encoded_images: A tensor containing a sequence of jpeg images.
    crop_window: The window to crop, as [y min, x min, height, width].
    image_size: The size to use to resize the images to after cropping, as
      [height, width].

  Returns:
    Video encoded as [T, image_size, image_size, C], where the time is the
    leading dimension.
  """
  video = decode_crop_images(jpeg_encoded_images, crop_window)
  return tf.image.resize(video, image_size)


def decode_crop_images(jpeg_encoded_images: tf.Tensor,
                       crop_window: tf.Tensor) -> tf.Tensor:
  """Given a crop window, decode the input tensors.

  Args:
    jpeg_encoded_images: A tensor containing jpeg images.
    crop_window: [row min, col min, row max, col max] the window to crop.

  Returns:
    Video encoded as [T, H, W, C], where the time is the leading dimension.
  """

  return tf.map_fn(
      lambda x: _decode_and_crop(x, crop_window),
      jpeg_encoded_images,
      fn_output_signature=tf.uint8)


def decode_resize_crop_images(jpeg_encoded_images: tf.Tensor, *,
                              initial_resize: int,
                              center_crop_size: int) -> tf.Tensor:
  """Decode, resize minimum and then center crop a sequence of images.

  Args:
    jpeg_encoded_images: A tensor containing jpeg images.
    initial_resize: First, resize the smallest edge of the images to be exactly
      this value.
    center_crop_size: Once the initial resize is complete,

  Returns:
    Video encoded as [T, H, W, C], where the time is the leading dimension.
  """

  video = tf.map_fn(
      tf.io.decode_jpeg, jpeg_encoded_images, fn_output_signature=tf.uint8)

  return _resize_min_and_crop(
      video, initial_resize=initial_resize, center_crop_size=center_crop_size)


def _resize_min_and_crop(video, *, initial_resize: int,
                         center_crop_size: int) -> tf.Tensor:
  """Resize the minimum side and center crop to given side.

  Args:
    video: The video to crop.
    initial_resize: First, resize the smallest edge of the images to be exactly
      this value.
    center_crop_size: Once the initial resize is complete,

  Returns:
    The cropped video.
  """

  video = resize_min(video, initial_resize)
  shape = tf.shape(video)
  height = shape[1]
  width = shape[2]

  offset_h = tf.cast((height - center_crop_size) / 2, tf.int32)
  offset_w = tf.cast((width - center_crop_size) / 2, tf.int32)

  return video[:, offset_h:offset_h + center_crop_size,
               offset_w:offset_w + center_crop_size, :]


def resize_min(video: tf.Tensor, shortest_edge: int) -> tf.Tensor:
  """Given a video, resize the smallest side to a given value.

  Args:
    video: A video as [T, H, W, 3].
    shortest_edge: The result will be resized so that the shortest edge matches
      this value.

  Returns:
    A video [T, H', W', 3], where min(H', W') = shortest_edge.
  """
  shape = tf.shape(video)
  input_h = shape[1]
  input_w = shape[2]

  output_h = tf.maximum(shortest_edge, (input_h * shortest_edge) // input_w)
  output_w = tf.maximum(shortest_edge, (input_w * shortest_edge) // input_h)

  def resize_fn():
    result = tf.image.resize(
        video, (output_h, output_w), method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(result, video.dtype)

  should_resize = tf.math.logical_or(
      tf.not_equal(input_w, output_w), tf.not_equal(input_h, output_h))

  return tf.cond(pred=should_resize, true_fn=resize_fn, false_fn=lambda: video)


def _decode_and_crop(jpeg_encoded_image: tf.Tensor,
                     crop_window: tf.Tensor) -> tf.Tensor:
  """Decode jpeg using to crop window.

  Args:
    jpeg_encoded_image: Tensor containing encoded jpeg.
    crop_window: [row min, col min, row max, col max] the window to crop.

  Returns:
    Decoded image [H, W, C]
  """

  return tf.image.decode_and_crop_jpeg(
      jpeg_encoded_image, crop_window, channels=3)


def _sample_crop_window(image_shape: tf.Tensor,
                        min_object_covered: float,
                        aspect_ratio_range: Tuple[float, float],
                        area_range: Tuple[float, float],
                        max_attempts: int,
                        seed: Optional[int] = None) -> tf.Tensor:
  """Sample a crop_window to be used for cropping.

  If the sampler fails to find a solution, the full imgae will be returned.

  Args:
    image_shape: The shape of the image to sample [height, width, channels].
    min_object_covered: The minimum amount of the image to cover.
    aspect_ratio_range: The range of aspect ratios of the result.
    area_range: The range of areas for the sampled boxes.
    max_attempts: The number of attempts to the sampler should make before
      failing.
    seed: The seed to feed to the random number generator.

  Returns:
    A crop window [min y, min x, height, width]. If the sampler fails,
    the resulting crop will be the full image.
  """

  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      image_shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True,
      seed=seed)

  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  return tf.stack([offset_y, offset_x, target_height, target_width])
