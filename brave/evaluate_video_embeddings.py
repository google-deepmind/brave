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

"""A runnable program to evaluate video embeddings.

Given a model checkpoint, and the location of the shards for a dataset,
computes the performance of the Brave video embeddings. This code
may be used to evaluate both UCF101 and HMDB51, as long as they are both
given in the appropriate input format. The only hyperparameter to this program
is the svm_regularization constant, which can impact the performance of the
linear classification.
"""

import glob
import json

from absl import app
from absl import flags
import chex
import jax
import numpy as np
import tensorflow as tf

from brave.datasets import datasets
from brave.evaluate import evaluate_video_embedding
from brave.models.brave import brave

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_path', None, 'Checkpoint to evaluate.')
flags.DEFINE_integer('batch_size', None, 'The size of the batches to use.')

# Hyperparameters
flags.DEFINE_float('svm_regularization', None, 'Regularization constant.')

# Datasets
flags.DEFINE_string('train_dataset_shards', None,
                    'Glob pattern for train shards.')
flags.DEFINE_string('test_dataset_shards', None,
                    'Glob pattern for test shards.')

# Transformations to apply to video before running network.
flags.DEFINE_integer('num_video_frames', 32, 'Number of frames in eval videos.')
flags.DEFINE_integer('video_step', 2, 'The step to use in the eval videos.')
flags.DEFINE_integer('image_size', 224, 'The size of the video to evaluate.')


def main(_):
  checkpoint_path = FLAGS.checkpoint_path

  train_shards = glob.glob(FLAGS.train_dataset_shards)
  test_shards = glob.glob(FLAGS.test_dataset_shards)

  video_config = evaluate_video_embedding.VideoConfig(
      num_frames=FLAGS.num_video_frames,
      image_size=FLAGS.image_size,
      video_step=FLAGS.video_step,
  )

  video_embedding_fn = _video_embedding(checkpoint_path)

  results = evaluate_video_embedding.evaluate_video_embedding(
      train_dataset_shards=train_shards,
      test_dataset_shards=test_shards,
      embedding_fn=video_embedding_fn,
      config=video_config,
      svm_regularization=FLAGS.svm_regularization,
      batch_size=FLAGS.batch_size)

  results_dct = dict(
      top_1_train=results.train.top_one_accuracy,
      top_5_train=results.train.top_five_accuracy,
      top_1_test=results.test.top_one_accuracy,
      top_5_test=results.test.top_five_accuracy,
  )

  # Write the results to stdout in a way that can be used as input to other
  # programs.
  print(json.dumps(results_dct))


def _video_embedding(checkpoint_path: str):
  """Load the video embedding for the BraVe model to evaluate."""

  checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
  params = checkpoint['params']
  state = checkpoint['state']
  brave_config_dct = checkpoint['config']

  brave_config = brave.BraveConfig(**brave_config_dct)
  model = brave.get_model(brave_config)

  @jax.jit
  def embedding_fn(view: datasets.View) -> chex.Array:
    narrow_forward_fn = model.forward_fns['narrow_video']
    embedding, _ = narrow_forward_fn(params, state, None, view, False)
    return embedding

  def synchronous_embedding_fn(view: datasets.View) -> chex.Array:
    # jax.jit causes the above function to be executed lazily, but we want
    # to force the computation to happen synchronously.
    return jax.device_get(embedding_fn(view))

  return synchronous_embedding_fn


if __name__ == '__main__':
  try:
    tf.config.set_visible_devices([], 'GPU')  # Prevent TF from using the GPU.
  except tf.errors.NotFoundError:
    pass

  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('batch_size')
  flags.mark_flag_as_required('train_dataset_shards')
  flags.mark_flag_as_required('test_dataset_shards')
  flags.mark_flag_as_required('svm_regularization')

  app.run(main)
