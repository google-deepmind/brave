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

"""The functions for computing gradient updates."""

from typing import Callable, NamedTuple, Sequence

import chex
import haiku as hk
import jax
import optax

from brave.datasets import datasets
from brave.models import embedding_model


class ModelUpdates(NamedTuple):
  params: hk.Params
  state: hk.State
  opt_state: optax.OptState
  scalars: embedding_model.Scalars


UpdateFn = Callable[
    [chex.PRNGKey, datasets.MiniBatch, hk.Params, hk.State, optax.OptState],
    ModelUpdates]


def build_update_fn(optimizer: optax.GradientTransformation,
                    loss_fn: embedding_model.LossFn) -> UpdateFn:
  """Returns a function for computing model updates.

  Args:
    optimizer: The optimizer to use e.g. the result of optax.sgd(...).
    loss_fn: An instance of the loss function, pmapped across all devices.

  Returns:
    A callable function that takes one step in the optimization problem using
    the gradients of the loss computed by the model loss function.
  """

  def update_fn(rng: chex.PRNGKey, minibatch: datasets.MiniBatch,
                params: hk.Params, state: hk.State,
                opt_state: optax.OptState) -> ModelUpdates:

    grad_fn = jax.grad(loss_fn, has_aux=True)
    grad, (state, scalars) = grad_fn(params, state, rng, minibatch)

    grad = jax.lax.pmean(grad, axis_name='i')
    scalars = jax.lax.pmean(scalars, axis_name='i')

    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return ModelUpdates(params, state, opt_state, scalars)

  return update_fn


def get_batch_dims(global_batch_size: int, device_count: int,
                   local_device_count: int) -> Sequence[int]:
  """Compute the batch dims for this host.

  The global_batch_size is the number of data samples that are optimized over
  in one step of the optimization. This value must be split up so that each
  individual device gets some share of the batch.

  When running with multiple devices, there may be multiple hosts, each
  with multiple local devices. Each host has a local copy of the program, and
  runs a local copy of the code. Each host must therefore use a batch size
  so that when all of the hosts run together, the total number of batched
  elements matches the global batch size. We do this by splitting up the global
  batch size evenly amongst all devices, and setting the batch size per host
  to the number of host devices times the device batch size.

  Args:
    global_batch_size: The target total batch size per optimization step.
    device_count: The total number of devices sharing computation per step.
    local_device_count: The number of devices available on the current host.

  Returns:
    The batch dimensions to use on the currently running host.
  """
  per_device_batch_size, remainder = divmod(global_batch_size, device_count)
  if remainder:
    raise ValueError(
        f'Cannot split batch of {global_batch_size} evenly across {local_device_count} devices.'
    )

  host_batch_dims = (local_device_count, per_device_batch_size)
  return host_batch_dims
