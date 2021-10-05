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

"""Define a re-usable container for multimodal embedding models."""

from typing import Dict, Tuple, Callable

import chex
import haiku as hk
import tensorflow as tf

from brave.datasets import datasets

Loss = chex.Array
Scalars = Dict[str, chex.Array]
GlobalStep = int
EvalMode = str

InitFn = Callable[[chex.PRNGKey], Tuple[hk.Params, hk.State]]

LossFn = Callable[[hk.Params, hk.State, chex.PRNGKey, datasets.MiniBatch],
                  Tuple[Loss, Tuple[hk.State, Scalars]]]

ForwardFn = Callable[[hk.Params, hk.State, chex.PRNGKey, datasets.View, bool],
                     Tuple[chex.Array, hk.State]]

DatasetBuilderFn = Callable[[], tf.data.Dataset]

EvaluateFn = Callable[[GlobalStep, EvalMode, hk.Params, hk.State],
                      Dict[str, chex.Array]]


@chex.dataclass
class MultimodalEmbeddingModel:
  """A trainable multimodal embedding model.

  Attributes:
    init_fn: The init function may be called to initialize parameters and state
      for the model. The resulting parameters may be used for all other
      functions returned in this dataclass.
    forward_fns: A mapping giving the (named) embedding functions trained by the
      model.
    loss_fn: A function to compute training loss given a train batch.
    evaluate_fn: A function taking the global step, the "evaluation mode" (a
      user-defined string taken from the Jaxline mode, which has been added to
      support running multiple parallel Jaxline evaluation processes), and
      returning a dictionary of metrics that may be published directly.
    train_dataset_builder_fn: A callable returning the train dataset. he dataset
      must be an iterable over datasets.MiniBatch, each minibatch must be
      structured to support the specific embedding model being trained. Note
      that the batch dimension is added by the trainer, so the dataset should
      return single examples, rather than batches.
  """
  init_fn: InitFn
  forward_fns: Dict[str, ForwardFn]
  loss_fn: LossFn
  evaluate_fn: EvaluateFn
  train_dataset_builder_fn: DatasetBuilderFn
