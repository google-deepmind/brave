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

"""Config for training Brave experiment."""

import ml_collections


def get_experiment_config() -> ml_collections.ConfigDict:
  """The config for the BraVe model`."""

  return ml_collections.ConfigDict({
      'global_batch_size': 512,
      'model_name': 'brave',
      'eval_modes': ['eval'],
      'model': {
          'image_size_broad': 112,
          'num_frames_broad': 64,
          'step_broad': 4,
          'include_video_in_broad_view': True,
          'include_audio_in_broad_view': True,
          'image_size_narrow': 224,
          'num_frames_narrow': 16,
          'step_narrow': 2,
          'output_dims': 128,
          'tsm_resnet_width_multiplier': 1,
          'num_spectrogram_bins': 80,
          'fft_step': 160,
          'dataset_shards': [],  # Set this to train the model.
          'input_audio_sample_rate': 48_000,
          'input_video_sample_rate': 25.0,
      },
      'optimizer': {
          'optimizer_name': 'adam',
          'weight_decay': 1e-2,
          'optimizer_kwargs': {},
          'scheduler_name': 'cosine_decay',
          'scheduler_kwargs': {
              'init_value': 0.0,
              'peak_value': 2e-3,
              'warmup_steps': 5000,
              'decay_steps': 300_000,
              'end_value': 0.0,
          },
          'scale_learning_rate_by_regex': [],
      },
  })
