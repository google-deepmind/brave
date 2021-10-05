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

"""Package providing functions for evaluation."""

from typing import Callable, Iterable, NamedTuple, Tuple

from absl import logging
import chex
import numpy as np
import sklearn
from sklearn import preprocessing
import sklearn.svm

from brave.datasets import datasets

DEFAULT_BATCH_SIZE = 1
DEFAULT_LOG_INTERVAL = 100
DEFAULT_REGULARIZATION_PARAMETER = 1.0

EmbeddingFn = Callable[[datasets.View], chex.Array]


@chex.dataclass
class ClassificationResults:
  """Classification evaluation results.

  Attributes:
    top_one_accuracy: How often most confident predition is correct in the range
      [0.0, 1.0].
    top_five_accuracy: How often the correct result is in the top five
      predictions in the range [0.0, 1.0].
  """
  top_one_accuracy: float
  top_five_accuracy: float


class EvaluationResults(NamedTuple):
  test: ClassificationResults
  train: ClassificationResults


def linear_svm_classifier(
    train_dataset: Iterable[datasets.MiniBatch],
    test_dataset: Iterable[datasets.MiniBatch],
    embedding_fn: EmbeddingFn,
    test_predictions_group_size: int = 1,
    svm_regularization: float = DEFAULT_REGULARIZATION_PARAMETER,
) -> EvaluationResults:
  """Evaluate the given embedding function for a classification dataset.

  Args:
    train_dataset: The dataset to fit the linear model (must contain a view
      called 'default').
    test_dataset: The dataset to test the linear model accuracy with (must
      contain a view called 'default').
    embedding_fn: The embedding maps the 'default' view in the dataset batches
      to an embedding, which is used as the space for the classifier.
      test_predictions_group_size: In some evaluation regimes, we break up the
        test data into many clips. For example, taking 10 temporal crops and 3
        spatial crops will result in each sample in the test set being split
        into 30 separate examples. When evaluating, we then wish to regroup
        these together and then average the prediction in order to compute the
        correct metrics. This requires that the dataset be in the correct order,
        so that adjacent samples in the test dataset belong to the same group.
        As a basic check we ensure that the labels for each of the test samples
        in a group have the same label.
    svm_regularization: The regularization constant to use in the SVM. Please
      see the accompanying paper for more information on selecting this value
      correctly.

  Returns:
    The accuracy achieved by the model / embedding combination
  """

  logging.info('Computing train embeddings.')
  train_embeddings, train_labels = _compute_embeddings(train_dataset,
                                                       embedding_fn)
  logging.info('Computed %d train embeddings.', train_embeddings.shape[0])

  logging.info('Computing test embeddings.')
  test_embeddings, test_labels = _compute_embeddings(test_dataset, embedding_fn)
  logging.info('Computed %d test embeddings.', test_embeddings.shape[0])

  logging.info('Learning a rescaler.')
  scaler = preprocessing.StandardScaler().fit(train_embeddings)

  logging.info('Rescaling features.')
  train_embeddings = scaler.transform(train_embeddings)
  test_embeddings = scaler.transform(test_embeddings)

  logging.info('Fitting an SVM with regularization %f.', svm_regularization)
  classifier = sklearn.svm.LinearSVC(C=svm_regularization)
  classifier.fit(train_embeddings, train_labels)

  logging.info('Computing predictions.')
  train_predictions = classifier.decision_function(train_embeddings)
  test_predictions = classifier.decision_function(test_embeddings)

  logging.info('Average over groups of size: %d.', test_predictions_group_size)
  test_predictions, test_labels = _average_test_predictions_by_group(
      test_predictions_group_size, test_predictions, test_labels)

  logging.info('Computing metrics.')
  return EvaluationResults(
      test=_compute_accuracy_metrics(test_labels, test_predictions),
      train=_compute_accuracy_metrics(train_labels, train_predictions),
  )


def _compute_embeddings(
    dataset: Iterable[datasets.MiniBatch],
    embedding_fn: EmbeddingFn) -> Tuple[chex.Array, chex.Array]:
  """Compute embeddings and labels for the given embedding function."""

  embeddings, labels = [], []

  for i, batch in enumerate(dataset):

    if i % DEFAULT_LOG_INTERVAL == 0:
      logging.info('Completed %d embedding batches.', i)

    if 'default' not in batch.views:
      raise ValueError(
          f'View named `default` not found, but is required. Got {batch.views.keys()}.'
      )

    view = batch.views['default']
    if view.labels is None:
      raise ValueError('Labels must be present for evaluation runs.')

    embeddings.append(embedding_fn(view))
    labels.append(view.labels)

  return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


def _compute_accuracy_metrics(labels: chex.Array,
                              predictions: chex.Array) -> ClassificationResults:
  """Compute accuracy metrics."""

  sorted_predictions = np.argsort(predictions, axis=1)
  assert len(labels.shape) == len(sorted_predictions.shape) == 2

  top1_predictions = sorted_predictions[:, -1:]
  top5_predictions = sorted_predictions[:, -5:]

  return ClassificationResults(
      top_one_accuracy=np.mean(top1_predictions == labels),
      top_five_accuracy=np.mean(np.max(top5_predictions == labels, 1)),
  )


def _average_test_predictions_by_group(
    group_size: int, predictions: chex.Array,
    labels: chex.Array) -> Tuple[chex.Array, chex.Array]:
  """Average contiguous predictions together."""

  if predictions.shape[0] % group_size != 0:
    raise ValueError('Predictions must be divisible by group size.')

  predictions = predictions.reshape((-1, group_size) +
                                    tuple(predictions.shape[1:]))
  labels = labels.reshape((-1, group_size) + tuple(labels.shape[1:]))
  averaged_predictions = predictions.mean(axis=1)

  # The labels in each group should be identical, an easy way to check this
  # is that the min and max are identical.
  labels_min = labels.min(axis=1)
  labels_max = labels.max(axis=1)
  np.testing.assert_equal(labels_min, labels_max)

  return averaged_predictions, labels_min
