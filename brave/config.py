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

"""Configuration for the Brave experiment."""

import glob
from jaxline import base_config as jaxline_base_config
import ml_collections

from brave.models.brave import config as brave_config


def get_config() -> ml_collections.ConfigDict:
  """Get the experiment config."""

  config = jaxline_base_config.get_base_config()

  config.checkpoint_dir = '/tmp/jaxline/brave'
  config.train_checkpoint_all_hosts = False
  config.training_steps = 300_000
  config.log_tensors_interval = 60
  config.save_checkpoint_interval = 600
  config.eval_specific_checkpoint_dir = ''
  config.best_model_eval_metric = 'multiple_of_saving_period'

  config.experiment_kwargs.experiment_name = 'brave'
  config.experiment_kwargs.config = brave_config.get_experiment_config()
  config.eval_modes = config.experiment_kwargs.config.eval_modes

  # Fill in this to set the training shards for training.
  config.experiment_kwargs.config.model.dataset_shards = glob.glob(
      '<path/to/train/shards/*.tfrecord>')

  config.lock()
  return config
