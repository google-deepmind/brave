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

"""Tests for data augmentations."""

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from brave.datasets import augmentations
from brave.datasets import datasets


class AugmentationsTest(parameterized.TestCase):

  def test_normalize_video(self):
    view = _view_fixture()
    result = augmentations.normalize_video(view)
    self.assertEqual(result.video.shape, (4, 8, 8, 3))
    self.assertAlmostEqual(result.video[0, 0, 0, 0], 0.0)

  def test_random_color_augment_video(self):
    view = _view_fixture()
    result = augmentations.random_color_augment_video(
        view, prob_color_augment=1.0, prob_color_drop=1.0)
    self.assertEqual(result.video.shape, (4, 8, 8, 3))

  def test_gaussian_blur(self):
    view = _view_fixture()
    result = augmentations.random_gaussian_blur_video(
        view, kernel_size=3, sigma_range=(1.0, 1.0))
    self.assertEqual(result.video.shape, (4, 8, 8, 3))

  def test_random_horizontal_flip_video(self):
    view = _view_fixture()
    result = augmentations.random_horizontal_flip_video(view)
    self.assertEqual(result.video.shape, (4, 8, 8, 3))

  def test_random_convolve_video(self):
    view = _view_fixture()
    result = augmentations.random_convolve_video(view)
    self.assertEqual(result.video.shape, (4, 8, 8, 3))


def _view_fixture() -> datasets.View:
  return datasets.View(video=tf.zeros([4, 8, 8, 3]), audio=None, labels=None)


if __name__ == '__main__':
  absltest.main()
