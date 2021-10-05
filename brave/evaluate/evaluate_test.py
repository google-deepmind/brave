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

"""Tests for evaluate."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from brave.datasets import datasets
from brave.evaluate import evaluate


class EvaluateTest(parameterized.TestCase):

  def test_evaluate_with_linear_svm_classifier(self):
    train_dataset = _fake_dataset()

    def embedding_fn(view):
      batch_size = view.video.shape[0]
      return np.zeros([batch_size, 10])

    test_dataset = train_dataset
    result = evaluate.linear_svm_classifier(train_dataset, test_dataset,
                                            embedding_fn)

    self.assertAlmostEqual(result.test.top_one_accuracy, 0.33333333333)
    self.assertAlmostEqual(result.test.top_five_accuracy, 1.0)

  def test_compute_accuracy_metrics(self):

    labels = np.array([[0], [2], [1], [0]])
    predictions = np.array([
        [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # top-1 correct
        [0.10, 0.10, 0.80, 0.00, 0.00, 0.00],  # top-1 correct
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.85],  # top-5 correct
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.85],  # top-6 correct
    ])
    results = evaluate._compute_accuracy_metrics(labels, predictions)
    # Two our of four are correct
    self.assertEqual(results.top_one_accuracy, 2.0 / 4.0)
    self.assertEqual(results.top_five_accuracy, 3.0 / 4.0)

  def test_average_test_predictions_by_group(self):

    predictions = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # clip 1 example 1
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # clip 2 example 1
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # clip 1 example 2
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # clip 2 example 2
    ])

    labels = np.array([[0], [0], [1], [1]])

    avg_predictions, new_labels = evaluate._average_test_predictions_by_group(
        2, predictions, labels)

    expected_predictions = np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])
    expected_labels = np.array([[0], [1]])

    np.testing.assert_almost_equal(expected_predictions, avg_predictions)
    np.testing.assert_almost_equal(expected_labels, new_labels)

    with self.assertRaises(ValueError):
      # 4 is not divisible by 3.
      evaluate._average_test_predictions_by_group(3, predictions, labels)


def _fake_dataset():
  return [
      datasets.MiniBatch(
          views={
              'default':
                  datasets.View(
                      video=np.zeros([3, 2, 4, 4, 3]),
                      audio=None,
                      labels=np.array([[0], [1], [2]]),
                  )
          },)
  ]


if __name__ == '__main__':
  absltest.main()
