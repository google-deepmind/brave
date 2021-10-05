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

"""Test the experiment."""

import tempfile
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import chex
from jaxline import train

from brave import config
from brave import experiment
from brave.datasets import fixtures

FLAGS = flags.FLAGS

DEVICE_COUNT = 1
chex.set_n_cpu_devices(DEVICE_COUNT)


class ExperimentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Needed to make absl work with pytest.
    FLAGS.mark_as_parsed()

  def test_train(self):
    with chex.fake_pmap(), tempfile.TemporaryDirectory() as fixture_dir:
      shards = fixtures.write_tf_record_dataset_fixture(fixture_dir)
      cfg = _lightweight_brave_config(shards)

      train.train(
          experiment.Experiment,
          cfg,
          checkpointer=mock.Mock(),
          writer=mock.Mock())


def _lightweight_brave_config(shards):
  cfg = config.get_config()
  cfg.unlock()
  cfg.training_steps = 1

  # Do whatever we can to make running this test as fast as possible.
  # without changing the experiment too much from a real run.
  cfg.experiment_kwargs.config.global_batch_size = 2
  cfg.experiment_kwargs.config.model.include_video_in_broad_view = True
  cfg.experiment_kwargs.config.model.include_audio_in_broad_view = False
  cfg.experiment_kwargs.config.model.output_dims = 2
  cfg.experiment_kwargs.config.model.image_size_broad = 2
  cfg.experiment_kwargs.config.model.num_frames_broad = 2
  cfg.experiment_kwargs.config.model.image_size_narrow = 2
  cfg.experiment_kwargs.config.model.num_frames_narrow = 1
  cfg.experiment_kwargs.config.model.dataset_shards = shards

  cfg.experiment_kwargs.experiment_name = 'brave_test'

  print(cfg)
  cfg.lock()
  return cfg


if __name__ == '__main__':
  absltest.main()
