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

"""Tests for spectrograms."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from brave.datasets import spectrograms


class AudioProcessingTest(parameterized.TestCase):

  def test_log_mel_spectrogram(self):
    pcm = np.random.uniform(low=-1.0, high=1.0, size=(48_000 * 13,))
    pcm = tf.constant(pcm, dtype=tf.float32)

    expected_length = int(48_000 * 13 / 160)
    log_mel = spectrograms.pcm_to_log_mel_spectrogram(
        pcm, input_sample_rate=48_000, num_spectrogram_bins=80, fft_step=160)

    self.assertEqual((expected_length, 80), log_mel.shape)

  def test_batched_spectrogram(self):
    shape = (3, 5, 48_000)
    pcm = np.random.uniform(low=-1.0, high=1.0, size=shape)
    pcm = tf.constant(pcm, dtype=tf.float32)
    spectrogram = spectrograms.pcm_to_log_mel_spectrogram(
        pcm, input_sample_rate=48_000, num_spectrogram_bins=80, fft_step=160)

    self.assertEqual((3, 5, int(48_000 / 160), 80), spectrogram.shape)


if __name__ == '__main__':
  absltest.main()
