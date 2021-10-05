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

"""Implement the standard evaluation procedure for video embeddings."""

from typing import Sequence

import chex
import tensorflow as tf
import tensorflow_datasets as tfds

from brave.datasets import augmentations
from brave.datasets import datasets
from brave.datasets import media_sequences
from brave.evaluate import eval_datasets
from brave.evaluate import evaluate

DEFAULT_EVAL_BATCH_SIZE = 1
DEFAULT_EVAL_NUM_TRAIN_EPOCHS = 10
DEFAULT_TRAIN_MIN_CROP_WINDOW_AREA = 0.3
DEFAULT_TRAIN_MAX_CROP_WINDOW_AREA = 1.0
DEFAULT_TRAIN_MIN_CROP_WINDOW_ASPECT_RATIO = 0.5
DEFAULT_TRAIN_MAX_CROP_WINDOW_ASPECT_RATIO = 2.0

DEFAULT_TEST_INITIAL_RESIZE = 256
DEFAULT_TEST_NUM_TEMPORAL_CROPS = 10
DEFAULT_TEST_NUM_SPATIAL_CROPS = 3


@chex.dataclass
class VideoConfig:
  num_frames: int
  image_size: int
  video_step: int


def evaluate_video_embedding(
    train_dataset_shards: Sequence[str],
    test_dataset_shards: Sequence[str],
    embedding_fn: evaluate.EmbeddingFn,
    config: VideoConfig,
    svm_regularization: float,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    shard_reader: media_sequences.ShardReaderFn = media_sequences
    .tf_record_shard_reader,
) -> evaluate.EvaluationResults:
  """Standardized evaluation for embeddings."""

  train_ds = eval_datasets.random_sampling_dataset(
      train_dataset_shards,
      image_size=config.image_size,
      num_video_frames=config.num_frames,
      video_step=config.video_step,
      min_crop_window_area=DEFAULT_TRAIN_MIN_CROP_WINDOW_AREA,
      max_crop_window_area=DEFAULT_TRAIN_MAX_CROP_WINDOW_AREA,
      min_crop_window_aspect_ratio=DEFAULT_TRAIN_MIN_CROP_WINDOW_ASPECT_RATIO,
      max_crop_window_aspect_ratio=DEFAULT_TRAIN_MAX_CROP_WINDOW_ASPECT_RATIO,
      shuffle=True,
      shard_reader=shard_reader)

  train_ds = train_ds.map(_transform_train, num_parallel_calls=tf.data.AUTOTUNE)
  train_ds = train_ds.repeat(DEFAULT_EVAL_NUM_TRAIN_EPOCHS)
  train_ds = train_ds.batch(batch_size)
  train_ds = tfds.as_numpy(train_ds)

  test_ds = eval_datasets.multiple_crop_dataset(
      test_dataset_shards,
      num_temporal_crops=DEFAULT_TEST_NUM_TEMPORAL_CROPS,
      num_spatial_crops=DEFAULT_TEST_NUM_SPATIAL_CROPS,
      num_video_frames=config.num_frames,
      video_step=config.video_step,
      initial_resize=DEFAULT_TEST_INITIAL_RESIZE,
      center_crop_size=config.image_size,
      shuffle=False,
      shard_reader=shard_reader)

  test_ds = test_ds.map(_transform_test, num_parallel_calls=tf.data.AUTOTUNE)
  test_ds = test_ds.batch(batch_size)
  test_ds = tfds.as_numpy(test_ds)

  group_size = DEFAULT_TEST_NUM_TEMPORAL_CROPS * DEFAULT_TEST_NUM_SPATIAL_CROPS

  return evaluate.linear_svm_classifier(
      train_ds,
      test_ds,
      embedding_fn,
      test_predictions_group_size=group_size,
      svm_regularization=svm_regularization)


def _transform_train(batch: datasets.MiniBatch) -> datasets.MiniBatch:
  """Transform the train set."""

  def augment(view):
    view = augmentations.normalize_video(view)
    view = augmentations.random_horizontal_flip_video(view)
    view = augmentations.random_color_augment_video(
        view, prob_color_augment=0.8, prob_color_drop=0.2)
    return view

  return datasets.MiniBatch(views={
      view_name: augment(view) for view_name, view in batch.views.items()
  })


def _transform_test(batch: datasets.MiniBatch) -> datasets.MiniBatch:
  """Transform the test set."""
  return datasets.MiniBatch(
      views={
          view_name: augmentations.normalize_video(view)
          for view_name, view in batch.views.items()
      })
