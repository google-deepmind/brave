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

"""Video sampling tests."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from brave.datasets import video_sampling


class VideoSamplingTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(image_shape=(224, 300), padding=0, expected=(0, 38, 224, 224)),
      dict(image_shape=(300, 224), padding=0, expected=(38, 0, 224, 224)),
      dict(image_shape=(224, 300), padding=16, expected=(16, 54, 192, 192)),
      dict(image_shape=(300, 224), padding=16, expected=(54, 16, 192, 192)),
      dict(image_shape=(32 + 1, 32 + 1), padding=16, expected=(16, 16, 1, 1)),
  ])
  def test_center_crop(self, image_shape, padding, expected):
    image_shape = tf.constant(image_shape, dtype=tf.int32)
    bbox = video_sampling.pad_and_center_crop_window(image_shape, padding)
    np.testing.assert_allclose(expected, bbox.numpy())

  @parameterized.parameters([
      dict(
          image_shape=(224, 300, 3),
          min_area=0.5,
          max_area=1.0,
          min_aspect_ratio=1.0,
          max_aspect_ratio=1.0),
      dict(
          image_shape=(224, 224, 3),
          min_area=0.5,
          max_area=1.0,
          min_aspect_ratio=0.3,
          max_aspect_ratio=2.0),
      dict(
          image_shape=(100, 10, 3),
          min_area=0.001,
          max_area=1.0,
          min_aspect_ratio=0.1,
          max_aspect_ratio=10.0),
  ])
  def test_random_sample_crop_window(self, image_shape, min_area, max_area,
                                     min_aspect_ratio, max_aspect_ratio):

    windows = []

    for i in range(100):
      crop_window = video_sampling.random_sample_crop_window(
          tf.constant(image_shape),
          min_area=min_area,
          max_area=max_area,
          min_aspect_ratio=min_aspect_ratio,
          max_aspect_ratio=max_aspect_ratio,
          seed=i).numpy()
      windows.append(crop_window)

    # Test that we see plenty of variety in the samples.
    different_samples = set(tuple(window) for window in windows)
    assert len(different_samples) > 50

    image_area = image_shape[0] * image_shape[1]

    sampled_min_area = min(w[2] * w[3] for w in windows)
    sampled_max_area = max(w[2] * w[3] for w in windows)
    sampled_min_aspect_ratio = min(w[3] / w[2] for w in windows)
    sampled_max_aspect_ratio = min(w[3] / w[2] for w in windows)

    self.assertLess(sampled_max_area / image_area, max_area + 1e-4)
    self.assertGreater(sampled_min_area / image_area, min_area - 1e-4)
    self.assertLess(sampled_max_aspect_ratio, max_aspect_ratio + 1e-4)
    self.assertGreater(sampled_min_aspect_ratio, min_aspect_ratio - 1e-4)

  def test_random_sample_crop_window_fall_back(self):
    # The sampler can't satisfy the given conditions, and will thus fallback
    # to a padded center crop. We check this by comparing with a padded
    # center crop gives the same result in this case.
    image_shape = tf.constant([224, 64, 3])
    crop_window = video_sampling.random_sample_crop_window(
        image_shape,
        min_area=0.5,
        max_area=1.0,
        min_aspect_ratio=100.0,
        max_aspect_ratio=200.0,
        seed=0)

    padded_center_crop = video_sampling.pad_and_center_crop_window(image_shape)
    np.testing.assert_allclose(padded_center_crop.numpy(), crop_window.numpy())

  def test_resize_min_and_crop(self):
    video = np.ones((3, 120, 240, 3))
    video[:, 50:70, 110:130, :] = 1.0

    cropped = video_sampling._resize_min_and_crop(
        video, initial_resize=60, center_crop_size=60)

    self.assertEqual(cropped.shape, (3, 60, 60, 3))

    expected = np.ones((3, 60, 60, 3))
    expected[:, 20:40, 20:40, :] = 1.0

    np.testing.assert_allclose(expected, cropped.numpy())


if __name__ == '__main__':
  absltest.main()
