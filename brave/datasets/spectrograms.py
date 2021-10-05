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

"""Package to compute spectrograms from PCM encoded audio."""

import tensorflow as tf

DEFAULT_FRAME_LENGTH = 320
DEFAULT_FFT_LENGTH = 320
DEFAULT_LOG_FACTOR = 10_000
DEFAULT_LOWER_EDGE_FREQUENCY_HZ = 80.0
DEFAULT_UPPER_EDGE_FREQUENCY_HZ = 7600.0


def pcm_to_log_mel_spectrogram(pcm: tf.Tensor, input_sample_rate: int,
                               num_spectrogram_bins: int, fft_step: int):
  """Compute log-mel spectrogram from raw audio.

  Args:
    pcm: The raw audio represented as PCM, with shape (BATCH_DIMS..., N) sampled
      at the sample rate `input_sample_rate`, and with zero or more batch
      dimensions.
    input_sample_rate: The samplerate of the input audio.
    num_spectrogram_bins: The number of bins in the output spectrogram.
    fft_step: The step size to use in the fft.

  Returns:
    The log-mel spectrogram of the raw audio, with shape
    (BATCH_DIMS... , N / `fft_step`, `num_bins`), where N is the number of
    samples in the input pcm.
  """
  stfts = tf.signal.stft(
      pcm,
      frame_length=DEFAULT_FRAME_LENGTH,
      frame_step=fft_step,
      fft_length=DEFAULT_FFT_LENGTH,
      window_fn=tf.signal.hann_window,
      pad_end=True)

  spectrograms = tf.abs(stfts)

  # Warp the linear scale spectrograms into the mel-scale.
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_spectrogram_bins, stfts.shape[-1], input_sample_rate,
      DEFAULT_LOWER_EDGE_FREQUENCY_HZ, DEFAULT_UPPER_EDGE_FREQUENCY_HZ)

  mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)

  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  # Using Sander's suggested alternative.
  log_mel_spectrograms = tf.math.log(1 + DEFAULT_LOG_FACTOR * mel_spectrograms)

  return log_mel_spectrograms
