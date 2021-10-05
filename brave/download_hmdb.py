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

"""Fetch and write HMDB51.

This binary can be run with

  python -m brave.download_hmdb --output_dir <desired/output/directory>

This will write a sharded tfrecords version of HMDB to the destination,
with each of the test and train splits separated out.

This binary requires that the ffmpeg and rar commands be callable as
subprocesses.

Note that a local temporary working directory is used to fetch and decompress
files at /tmp/hdmb. This can be set using the `work_dir` flag.
"""

import contextlib
import enum
import glob
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, NamedTuple, Sequence

from absl import app
from absl import flags
from absl import logging
import ffmpeg
import requests
import tensorflow as tf
import tqdm

DEFAULT_UNRAR_COMMAND = 'unrar'
DEFAULT_HMDB_SPLITS_URL = 'https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar'
DEFAULT_HMDB_DATA_URL = 'https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'
DEFAULT_WORK_DIR = os.path.join(tempfile.gettempdir(), 'hmdb')

DEFAULT_VIDEO_QUALITY = 1
DEFAULT_VIDEO_FRAMERATE = 25
DEFAULT_VIDEO_MIN_RESIZE = 256
DEFAULT_MAX_SAMPLE_LENGTH_SECONDS = 20.0

MD5_OF_SPLITS_FILE = '15e67781e70dcfbdce2d7dbb9b3344b5'
MD5_OF_VIDEOS_FILE = '517d6f1f19f215c45cdd4d25356be1fb'

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'work_dir', DEFAULT_WORK_DIR,
    'Temporary working directory for downloading and processing files')
flags.DEFINE_string('output_dir', None,
                    'Where to write the dataset shards at the end')
flags.DEFINE_string('unrar_command', DEFAULT_UNRAR_COMMAND,
                    'Path to call unrar')
flags.DEFINE_integer('num_shards', 10, 'Number of shards to write')


class Split(enum.Enum):
  UNUSED = 0
  TRAIN = 1
  TEST = 2


class HMDBVideo(NamedTuple):
  hmdb_split_index: int
  split: Split
  full_video_path: str
  action_name: str


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments')

  output_dir = FLAGS.output_dir
  os.makedirs(output_dir, exist_ok=True)

  if os.listdir(output_dir):
    logging.error('Output directory `%s` must be empty', output_dir)
    exit(1)

  work_dir = FLAGS.work_dir
  video_dir = os.path.join(work_dir, 'videos')

  os.makedirs(work_dir, exist_ok=True)
  os.makedirs(video_dir, exist_ok=True)

  logging.info('Using `%s` as a temporary work_dir', work_dir)

  _fetch_dataset_splits(work_dir)
  _fetch_and_decompress_videos(video_dir)

  videos = _get_all_hmdb_videos(work_dir, video_dir)
  _write_dataset(videos, output_dir, num_shards=FLAGS.num_shards)

  logging.info('Finished writing tfrecords dataset to %s', output_dir)


def _write_dataset(videos: Sequence[HMDBVideo], output_dir: str,
                   num_shards: int) -> None:
  """Write a tfrecords dataset for the given videos.

  Args:
    videos: The sequence of all videos in the dataset to write.
    output_dir: The destination where the dataset will be written.
    num_shards: The number of independent shards to write the dataset to.
  """
  for split_index in [1, 2, 3]:
    split_index_name = f'split_{split_index}'

    for split in [Split.TEST, Split.TRAIN]:
      split_name = 'test' if split == split.TEST else 'train'
      current_dir = os.path.join(output_dir, split_index_name, split_name)
      os.makedirs(current_dir, exist_ok=True)

      videos_to_write = [
          video for video in videos
          if video.hmdb_split_index == split_index and video.split == split
      ]

      logging.info('Writing %d videos to %d shards at %s', len(videos_to_write),
                   num_shards, current_dir)
      _write_shards(videos_to_write, current_dir, num_shards)


def _fetch_and_decompress_videos(video_dir: str) -> None:
  """Download and extract rarfiles containing videos.

  At the end of this command, all videos will be stored in the given
  video_dir. This function also leaves the original .rar files in place.

  Args:
    video_dir: The directory in the workspace where the video should be
      downloaded and extracted.
  """

  videos_rar_path = os.path.join(video_dir, 'hmdb51_org.rar')
  _download_data(DEFAULT_HMDB_DATA_URL, videos_rar_path)

  videos_md5 = md5(videos_rar_path)
  logging.info('MD5 of videos rar: %s', videos_md5)
  if videos_md5 != MD5_OF_VIDEOS_FILE:
    logging.warning(
        'MD5 did not match expected value (%s)'
        ' - this may cause this script to fail', MD5_OF_SPLITS_FILE)
  logging.info('Extracting rarfile `%s` in `%s`', videos_rar_path, video_dir)
  _extract_rar(videos_rar_path, video_dir)

  logging.info('Extracting video rar files')
  for path in tqdm.tqdm(os.listdir(video_dir)):
    if path == 'hmdb51_org.rar':
      continue
    _extract_rar(path, video_dir)


def _fetch_dataset_splits(work_dir: str) -> None:
  """Download the datafile containing the splits information.

  Args:
    work_dir: The location where the temporary files may be downloaded and
      decompressed.
  """

  splits_rar_path = os.path.join(work_dir, 'test_train_splits.rar')
  _download_data(DEFAULT_HMDB_SPLITS_URL, splits_rar_path)
  splits_md5 = md5(splits_rar_path)

  logging.info('MD5 of splits file: %s', splits_md5)
  if splits_md5 != MD5_OF_SPLITS_FILE:
    logging.warning(
        'MD5 did not match expected value (%s)'
        ' - this may cause this script to fail', MD5_OF_SPLITS_FILE)

  logging.info('Extracting rarfile `%s` in `%s`', splits_rar_path, work_dir)
  _extract_rar(splits_rar_path, work_dir)


def _write_shards(hmdb_videos: Sequence[HMDBVideo], output_dir: str,
                  num_shards: int) -> None:
  """Write tfrecord shards to the output dir.

  Args:
    hmdb_videos: The videos to write to the output directory.
    output_dir: The location where the shards will be written.
    num_shards: The number of shards to write.
  """

  shard_paths = [
      os.path.join(output_dir, f'hmdb51-{i:05d}-of-{num_shards:05d}')
      for i in range(num_shards)
  ]

  with contextlib.ExitStack() as context_manager:
    writers = [
        context_manager.enter_context(tf.io.TFRecordWriter(path))
        for path in shard_paths
    ]

    all_actions = set(video.action_name for video in hmdb_videos)
    action_lookup = {
        action_name: index
        for index, action_name in enumerate(sorted(all_actions))
    }

    for i, video in enumerate(tqdm.tqdm(hmdb_videos)):
      sequence = _create_sequence_example(video, action_lookup)
      writers[i % len(writers)].write(sequence.SerializeToString())


def _download_data(url: str, destination: str) -> None:
  """Fetch data from a url to the given destination.

  If the destination file is found to exist, no download will take place.
  Note that since this write is not atomic, if a download partially fails, then
  this might cause future runs to fail. Deleting all data in the work_dir will
  fix this.

  Args:
    url: The resource to fetch.
    destination: the full path where the output should be written.
  """

  if os.path.exists(destination):
    logging.info('Found data at `%s`, skipping download.', destination)
    return

  logging.info('Downloading from `%s` to `%s`', url, destination)
  with requests.get(url, stream=True) as r, open(destination, 'wb') as w:
    shutil.copyfileobj(r.raw, w)


def _extract_rar(filename: str, work_dir: str) -> None:
  _check_unrar_found()
  subprocess.call([FLAGS.unrar_command, 'e', '-y', '-idq', filename],
                  cwd=work_dir)


def _check_unrar_found() -> None:
  try:
    subprocess.call([FLAGS.unrar_command, '-idq'])
  except:
    raise RuntimeError(
        f'Failed to call unrar using command `{FLAGS.unrar_command}`. '
        'Unrar can be downlaoded at https://www.rarlab.com/download.htm.')


def _get_all_hmdb_videos(work_dir: str, video_dir: str) -> List[HMDBVideo]:
  """Extract splits data.

  Args:
    work_dir: The location containing the split txt files.
    video_dir: The location where the video data can be found.

  Returns:
    A list of HMDBVideo dataclasses containing information about each example
    in the dataset.
  """
  result = []

  for path in glob.glob(os.path.join(work_dir, '*txt')):
    match = re.search(r'^(.+)_test_split(\d)\.txt$', path)
    if not match:
      raise ValueError(f'Failed to parse path name: `{path}`')
    action_name = match.group(1)
    split_index = int(match.group(2))

    with open(path, 'r') as f:
      for line in f:
        line = line.strip()
        try:
          video_path, test_train_index = line.split(' ')
        except:
          raise ValueError(f'Failed to parse line `{line}`')

        if test_train_index == '0':
          split = Split.UNUSED
        elif test_train_index == '1':
          split = Split.TRAIN
        elif test_train_index == '2':
          split = Split.TEST
        else:
          raise ValueError(f'Unknown split `{test_train_index}`')

        result.append(
            HMDBVideo(
                hmdb_split_index=split_index,
                split=split,
                action_name=action_name,
                full_video_path=os.path.join(video_dir, video_path)))

  return result


def _create_sequence_example(
    video: HMDBVideo,
    label_to_label_index: Dict[str, int]) -> tf.train.SequenceExample:
  """Create a tf example using the conventions used by DMVR."""

  jpeg_encoded_images = _extract_jpeg_frames(video.full_video_path)

  def _jpeg_feature(buffer) -> tf.train.Feature:
    bytes_list = tf.train.BytesList(value=[buffer])
    return tf.train.Feature(bytes_list=bytes_list)

  def _label_feature(value: Sequence[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  video_feature = tf.train.FeatureList(
      feature=[_jpeg_feature(img) for img in jpeg_encoded_images])

  features = {'image/encoded': video_feature}

  label_index = label_to_label_index[video.action_name]
  context = tf.train.Features(feature={
      'clip/label/index': _label_feature([label_index]),
  })

  return tf.train.SequenceExample(
      context=context,
      feature_lists=tf.train.FeatureLists(feature_list=features))


def _extract_jpeg_frames(
    video_path: str,
    *,
    max_length: float = DEFAULT_MAX_SAMPLE_LENGTH_SECONDS,
    fps: int = DEFAULT_VIDEO_FRAMERATE,
    min_resize: int = DEFAULT_VIDEO_MIN_RESIZE) -> List[bytes]:
  """Extract list of encoded jpegs from video_path using ffmpeg."""

  jpeg_header = b'\xff\xd8'
  new_width = '(iw/min(iw,ih))*{}'.format(min_resize)

  # Note: the qscale parameter here is important for achieving good performance.
  cmd = (
      ffmpeg.input(video_path).trim(start=0, end=max_length).filter(
          'fps', fps=fps).filter('scale', new_width, -1).output(
              'pipe:',
              format='image2pipe',
              **{
                  'qscale:v': DEFAULT_VIDEO_QUALITY,
                  'qmin': DEFAULT_VIDEO_QUALITY,
                  'qmax': DEFAULT_VIDEO_QUALITY
              }))

  jpeg_bytes, _ = cmd.run(capture_stdout=True, quiet=True)
  jpeg_bytes = jpeg_bytes.split(jpeg_header)[1:]
  jpeg_bytes = map(lambda x: jpeg_header + x, jpeg_bytes)
  return list(jpeg_bytes)


def _set_context_int(key: str, value: int,
                     sequence: tf.train.SequenceExample) -> None:
  sequence.context.feature[key].int64_list.value[:] = (value,)


def _set_context_bytes(key: str, value: bytes,
                       sequence: tf.train.SequenceExample):
  sequence.context.feature[key].bytes_list.value[:] = (value,)


def _add_bytes_list(key: str, values: Sequence[bytes],
                    sequence: tf.train.SequenceExample) -> None:
  sequence.feature_lists.feature_list[key].feature.add(
  ).bytes_list.value[:] = values


def md5(path: str) -> str:
  """Compute an MD5 hash of the file at a given path."""
  hash_md5 = hashlib.md5()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(4096), b''):
      hash_md5.update(chunk)
  return hash_md5.hexdigest()


if __name__ == '__main__':
  flags.mark_flag_as_required('output_dir')
  app.run(main)
