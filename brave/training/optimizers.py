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

"""Optimizers for the trainer."""

import re
from typing import Any, Callable, Dict, Sequence, Tuple

import chex
import jax.numpy as jnp
import optax
import tree


@chex.dataclass
class OptimizerConfig:
  """Shared config for optimizers.

  Attributes:
    optimizer_name: The name of the optimizer (for example `lars`, `adam`).
    scheduler_name: The name of the scheduler (for example. `cosine_decay`).
    scheduler_kwargs: Kwargs to pass to the scheduler function.
    optimizer_kwargs: Kwargs to pass to the optimizer function.
    weight_decay: The constant to use for weight decay.
    scale_learning_rate_by_regex: A sequence of regular expressions to match
      module parameter paths, along with the rescaling to apply. This allows us
      to tune the learning rate for individual parameter blocks. For example
      `[(r'^.*predictor.*$', 10.0)] would match modules all modules containing
      the name 'predictor' and rescale the learning rate by a factor of 10.0.
  """
  optimizer_name: str
  optimizer_kwargs: Dict[str, Any]

  scheduler_name: str
  scheduler_kwargs: Dict[str, Any]

  weight_decay: float
  scale_learning_rate_by_regex: Sequence[Tuple[str, float]]


def exclude_bias_and_normalizers(params):

  def predicate(path: Tuple[Any], value: jnp.ndarray) -> jnp.ndarray:
    del value
    return path[-1] == 'b' or 'norm' in path[-2]

  return tree.map_structure_with_path(predicate, params)


def get_optimizer(
    config: OptimizerConfig,
    *,
    weight_decay_mask: Callable[[optax.Params],
                                optax.Params] = exclude_bias_and_normalizers,
    trust_ratio_mask: Callable[[optax.Params],
                               optax.Params] = exclude_bias_and_normalizers
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
  """Returns the optimizer.

  The function return the optimizer function for the model.

  Args:
    config: parameters for the optimizer and the scheduler.
    weight_decay_mask: A mask used to remove parameter blocks from weight decay.
    trust_ratio_mask: A mask used to remove parameter blocks from LARS trust
      ratio update.

  Returns:
    the corresponding `GradientTransformation`.
  """

  learning_rate_schedule = _get_learning_rate_schedule(config)

  if config.optimizer_name == 'adam':
    optimizer = optax.chain(
        _scale_by_module_name(config.scale_learning_rate_by_regex),
        optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=config.weight_decay,
            mask=weight_decay_mask,
            **config.optimizer_kwargs))
    return optimizer, learning_rate_schedule

  elif config.optimizer_name == 'lars':
    optimizer = optax.chain(
        _scale_by_module_name(config.scale_learning_rate_by_regex),
        optax.lars(
            learning_rate=learning_rate_schedule,
            weight_decay=config.weight_decay,
            weight_decay_mask=weight_decay_mask,
            trust_ratio_mask=trust_ratio_mask,
            **config.optimizer_kwargs))
    return optimizer, learning_rate_schedule

  else:
    raise ValueError(f'Unknown optimizer: {config.optimizer_name}.')


def _get_learning_rate_schedule(config: OptimizerConfig):
  if config.scheduler_name == 'cosine_decay':
    return optax.warmup_cosine_decay_schedule(**config.scheduler_kwargs)
  elif config.scheduler_name == 'piecewise_constant':
    return optax.piecewise_constant_schedule(**config.scheduler_kwargs)
  else:
    raise ValueError(f'Unknown scheduler: {config.scheduler_name}.')


class _ScaleByModuleNameState(optax.OptState):
  ...


def _scale_by_module_name(
    module_regex_and_scale: Sequence[Tuple[str, float]]
) -> optax.GradientTransformation:
  """An transformation that rescales the updates only of matching layers.

  Args:
    module_regex_and_scale: A sequence of pairs of regex pattern and scale.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  matchers = [
      (re.compile(pattern), scale) for pattern, scale in module_regex_and_scale
  ]

  def rescaler(path: Tuple[Any], value: jnp.ndarray) -> jnp.ndarray:
    path = '/'.join(path)
    rescaling = 1.0
    for matcher, scale in matchers:
      if matcher.match(path):
        rescaling *= scale
    return value * rescaling

  def init_fn(_):
    return _ScaleByModuleNameState()

  def update_fn(updates, state, params=None):
    del params
    return tree.map_structure_with_path(rescaler, updates), state

  return optax.GradientTransformation(init_fn, update_fn)
