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

"""A jaxline experiment to train the Brave model."""

import functools
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import chex
import jax
from jaxline import experiment
from jaxline import platform
from jaxline import utils
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds

from brave.models.brave import brave
from brave.training import optimizers
from brave.training import trainer

FLAGS = flags.FLAGS


@chex.dataclass
class ExperimentConfig:
  """The config class for the experiment.

  Attributes:
    model_name: The name of the model to initialize and train.
    global_batch_size: The size of the batches to take from the train dataset.
      This will be split amongst all of the devices.
    optimizer: The configuration to use for the optimizer.
    eval_modes: For each value in this sequence, a new evaluation process will
      be started, with the given mode. This allows running multiple parallel
      evaluation processes.
  """
  model_name: str
  global_batch_size: int
  model: ml_collections.ConfigDict
  optimizer: ml_collections.ConfigDict
  eval_modes: Sequence[str]


class Experiment(experiment.AbstractExperiment):
  """Experiment to train Brave."""

  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_state': 'state',
      '_opt_state': 'opt_state',
  }

  def __init__(self, mode: str, init_rng: chex.Array,
               config: ml_collections.ConfigDict, experiment_name: str):

    super().__init__(mode=mode, init_rng=init_rng)
    self._mode = mode
    self._init_rng = init_rng
    self._config = ExperimentConfig(**config)
    self._model = brave.get_model(brave.BraveConfig(**config.model))
    self._opt_state = None
    self._params = None
    self._state = None

    logging.info('Running experiment in mode %s.', mode)
    if mode == 'train':
      self._init_train_dataset()
      self._init_train()

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def _init_train(self):
    """Prepare the model for training."""

    # Note: We store a copy of the learning rate schedule for logging only.
    optimizer_config = optimizers.OptimizerConfig(**self._config.optimizer)
    optimizer, self._lr_schedule = optimizers.get_optimizer(optimizer_config)

    init_fn = jax.pmap(self._model.init_fn, axis_name='i')
    optimizer_init_fn = jax.pmap(optimizer.init, axis_name='i')
    broadcasted_key = utils.bcast_local_devices(self._init_rng)

    self._params, self._state = init_fn(broadcasted_key)
    self._opt_state = optimizer_init_fn(self._params)

    self._update_fn = jax.pmap(
        trainer.build_update_fn(optimizer, self._model.loss_fn),
        axis_name='i',
        donate_argnums=(1, 2, 3, 4))

  def _init_train_dataset(self):
    batch_dims = trainer.get_batch_dims(self._config.global_batch_size,
                                        jax.device_count(),
                                        jax.local_device_count())
    logging.info('This host batch dimensions: %s.', batch_dims)

    ds = self._model.train_dataset_builder_fn()
    for batch_size in reversed(batch_dims):
      ds = ds.batch(batch_size, drop_remainder=True)

    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = tfds.as_numpy(ds)

    self._train_data_iter = utils.py_prefetch(lambda: iter(ds))

  def step(self, *, rng, writer, global_step):
    """Perform one step of the optimization."""

    del writer

    batch = next(self._train_data_iter)
    logging.log_every_n(logging.INFO, 'Batch shape: %s', 10, batch)

    updates = self._update_fn(rng, batch, self._params, self._state,
                              self._opt_state)
    self._params = updates.params
    self._state = updates.state
    self._opt_state = updates.opt_state

    scalars = updates.scalars
    scalars['learning_rate'] = self._lr_schedule(global_step)

    return utils.get_first(scalars)

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, *, rng, writer, global_step):
    params = utils.get_first(self._params)
    state = utils.get_first(self._state)
    return self._model.evaluate_fn(global_step, self._mode, params, state)


if __name__ == '__main__':
  flags.mark_flag_as_required('config')

  try:
    tf.config.set_visible_devices([], 'GPU')  # Prevent TF from using the GPU.
  except tf.errors.NotFoundError:
    pass

  app.run(functools.partial(platform.main, Experiment))
