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

"""Implement test fixtures for testing datasets."""

import os
from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf

DEFAULT_AUDIO_SAMPLES = int(10.376 * 48_000)


def write_tf_record_dataset_fixture(path: str) -> List[str]:
  """Write a tfrecord dataset to the given path."""

  records = [record for k, record in sorted(_make_fixture().items())]

  shard_1 = os.path.join(path, 'shard1.tfrecords')
  shard_2 = os.path.join(path, 'shard2.tfrecords')

  with tf.io.TFRecordWriter(shard_1) as w:
    for record in records[:2]:
      w.write(record)

  with tf.io.TFRecordWriter(shard_2) as w:
    for record in records[2:]:
      w.write(record)

  return [shard_1, shard_2]


def _make_fixture() -> Dict[str, bytes]:
  return {
      'sequence_1': _fake_sequence_1(0).SerializeToString(),
      'sequence_2': _fake_sequence_1(1).SerializeToString(),
      'sequence_3': _fake_sequence_1(2).SerializeToString(),
  }


def _fake_sequence_1(label: int) -> tf.train.SequenceExample:
  """The first test sequence."""

  img = np.zeros((224, 300, 3))

  video = tf.train.FeatureList(feature=[
      _jpeg_feature(img),
      _jpeg_feature(img),
      _jpeg_feature(img),
      _jpeg_feature(img),
      _jpeg_feature(img),
  ])

  audio = _audio_feature(np.zeros((DEFAULT_AUDIO_SAMPLES,)))

  features = {
      'image/encoded': video,
      'WAVEFORM/feature/floats': audio,
  }

  context = tf.train.Features(feature={
      'clip/label/index': _label_feature([label]),
  })

  return tf.train.SequenceExample(
      context=context,
      feature_lists=tf.train.FeatureLists(feature_list=features))


def _jpeg_feature(img: np.ndarray) -> tf.train.Feature:
  buffer = tf.io.encode_jpeg(img).numpy()
  bytes_list = tf.train.BytesList(value=[buffer])
  return tf.train.Feature(bytes_list=bytes_list)


def _audio_feature(value: np.ndarray) -> tf.train.Feature:
  return tf.train.FeatureList(
      feature=[tf.train.Feature(float_list=tf.train.FloatList(value=value))])


def _label_feature(value: Sequence[int]) -> tf.train.Feature:
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
