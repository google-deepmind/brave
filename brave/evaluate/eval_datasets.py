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

"""Implement datasets specifically for use in evaluation."""

from typing import Dict, Sequence

import tensorflow as tf

from brave.datasets import datasets as brave_datasets
from brave.datasets import media_sequences
from brave.datasets import sampling
from brave.datasets import time_sampling
from brave.datasets import video_sampling


def random_sampling_dataset(
    shards: Sequence[str],
    *,
    num_video_frames: int,
    video_step: int,
    image_size: int,
    min_crop_window_area: float,
    max_crop_window_area: float,
    min_crop_window_aspect_ratio: float,
    max_crop_window_aspect_ratio: float,
    shuffle: bool = False,
    shard_reader: media_sequences.ShardReaderFn = media_sequences
    .tf_record_shard_reader
) -> tf.data.Dataset:
  """A dataset that uses random cropping.

  For each clip in the underlying data, we return a single cropped sample, which
  has been sampled at random. We uniformly sample a start location of the video,
  and uniformly sample a crop window of shape `crop_size` X `crop_size`.

  Args:
    shards: The paths containing the dataset shards to read.
    num_video_frames: The required number of frames in the resulting videos.
    video_step: The gap between the frames sampled into the output video.
    image_size: Returned videos will have resolution `image_size` X
      `image_size`.
    min_crop_window_area: The minimum area of the source clip that the random
      crop must occupy.
    max_crop_window_area: The maximum area of the source clip that the random
      crop may occupy.
    min_crop_window_aspect_ratio: The minimum aspect ratio that the sampled crop
      window is allowed to have (before resizing).
    max_crop_window_aspect_ratio: The maximum aspect ratio that the sampled crop
      window is allowed to have (before resizing).
    shuffle: Whther or not to shuffle the resulting dataset.
    shard_reader: The reader taking shard paths and returning a dataset over
      encoded records.

  Returns:
    A tensorflow dataset with no batch dimensions, containing a single view
    name 'default' containing the results of the random sampling.
  """

  def sampler(sequence):
    return _random_sampler(
        sequence, num_video_frames=num_video_frames, video_step=video_step)

  def decoder(sequences):
    return _random_cropping_decoder(
        sequences,
        image_size=image_size,
        min_crop_window_area=min_crop_window_area,
        max_crop_window_area=max_crop_window_area,
        min_crop_window_aspect_ratio=min_crop_window_aspect_ratio,
        max_crop_window_aspect_ratio=max_crop_window_aspect_ratio,
    )

  return brave_datasets.multi_view_dataset(
      shards,
      features=[
          media_sequences.FeatureKind.VIDEO, media_sequences.FeatureKind.LABELS
      ],
      view_sampler=sampler,
      view_decoder=decoder,
      shuffle=shuffle,
      shard_reader=shard_reader)


def multiple_crop_dataset(
    shards: Sequence[str],
    *,
    num_temporal_crops: int,
    num_spatial_crops: int,
    num_video_frames: int,
    video_step: int,
    initial_resize: int,
    center_crop_size: int,
    shuffle: bool,
    shard_reader: media_sequences.ShardReaderFn = media_sequences
    .tf_record_shard_reader
) -> tf.data.Dataset:
  """A dataset giving many deterministic crops batched together for each clip.

  Args:
    shards: The sharded paths for to the source data to load.
    num_temporal_crops: The number of temporal crops to perform. If this is one,
      then the crop starts at the beginning of the sequence. If it is two, the
      first crop starts at the beginning, and the second is as close to the end
      as possible. For values greater than two, the crops are evenly spaced
      between those two endpoints.
    num_spatial_crops: Crop the video horizontally in this number of crops. The
      sampling logic is the same as the temporal sampling.
    num_video_frames: The number of video frames in each returned clip.
    video_step: The gap between video frames as sampled from the source data.
    initial_resize: When reading raw data, the videos are first resized so that
      their shortest edge matches this value.
    center_crop_size: After the initial resize, crops of this size are sampled
      from the center of the video (note that when there are multiple spatial
      crops, these are sampled according to the logic given above.
    shuffle: Whether or not to shuffle the resulting data.
    shard_reader: The reader taking shards and returning a tf.data.Dataset over
      serialized records.

  Returns:
    A dataset where, for each clip in the source data, `num_spatial_crops` X
    `num_temporal_crops` individual samples are returned, each containing
    exaclty one clip.
  """

  def sampler(sequence):
    return {'default': sequence}

  def decoder(sequences):
    return _multi_crop_decoder(
        sequences,
        num_temporal_crops=num_temporal_crops,
        num_spatial_crops=num_spatial_crops,
        num_video_frames=num_video_frames,
        video_step=video_step,
        initial_resize=initial_resize,
        center_crop_size=center_crop_size)

  ds = brave_datasets.multi_view_dataset(
      shards,
      features=[
          media_sequences.FeatureKind.VIDEO, media_sequences.FeatureKind.LABELS
      ],
      view_sampler=sampler,
      view_decoder=decoder,
      shuffle=shuffle,
      shard_reader=shard_reader)

  # The decoder above adds a batch dimension for each of the multiple crops.
  # For consistency with the other datasets, we now remove it.
  ds = ds.unbatch()

  return ds


def _random_sampler(
    sequence: media_sequences.EncodedSequence, num_video_frames: int,
    video_step: int) -> Dict[str, media_sequences.EncodedSequence]:
  """Random sample the given number of frames.

  Args:
    sequence: The sequence to sample from.
    num_video_frames: The number of frames to sample.
    video_step: The gap between frames as sampled from the sequence.

  Returns:
    A single sequence encoded as 'default'.
  """

  min_frames_required = (num_video_frames - 1) * video_step + 1
  sequence = media_sequences.extend_sequence(sequence, min_frames_required)
  result = time_sampling.random_sample_sequence_using_video(
      num_video_frames=num_video_frames,
      video_frame_step=video_step,
      sequence=sequence)

  return {'default': result.sequence}


def _random_cropping_decoder(
    sequences: Dict[str, media_sequences.EncodedSequence],
    *,
    image_size: int,
    min_crop_window_area: float,
    max_crop_window_area: float,
    min_crop_window_aspect_ratio: float,
    max_crop_window_aspect_ratio: float,
) -> Dict[str, brave_datasets.View]:
  """Randomly crop from an underlying sequence."""

  result = {}

  for view_name, sequence in sequences.items():
    image_shape = tf.image.extract_jpeg_shape(sequence.jpeg_encoded_images[0])

    crop_window = video_sampling.random_sample_crop_window(
        image_shape,
        min_area=min_crop_window_area,
        max_area=max_crop_window_area,
        min_aspect_ratio=min_crop_window_aspect_ratio,
        max_aspect_ratio=max_crop_window_aspect_ratio)

    video = video_sampling.decode_crop_resize_images(
        sequence.jpeg_encoded_images,
        crop_window,
        image_size=(image_size, image_size))

    result[view_name] = brave_datasets.View(
        video=video, labels=sequence.labels, audio=None)

  return result


def _multi_crop_decoder(
    sequences: Dict[str, media_sequences.EncodedSequence],
    num_temporal_crops: int, num_spatial_crops: int, num_video_frames: int,
    video_step: int, initial_resize: int,
    center_crop_size: int) -> Dict[str, brave_datasets.View]:
  """Sample a sequence multiple times, spatially and temporally."""

  result = {}

  for view_name, sequence in sequences.items():
    result[view_name] = _multi_crop_view_decoder(sequence, num_temporal_crops,
                                                 num_spatial_crops,
                                                 num_video_frames, video_step,
                                                 initial_resize,
                                                 center_crop_size)

  return result


def _multi_crop_view_decoder(sequence, num_temporal_crops, num_spatial_crops,
                             num_video_frames, video_step, initial_resize,
                             center_crop_size) -> brave_datasets.View:
  """Extract multiple temporal and spatial crops from a sequence.

  Args:
    sequence: The sequence to sample from.
    num_temporal_crops: The number of temporal crops to take.
    num_spatial_crops: The number of spatial crops to take. These crops are
      currently always taken horizontally.
    num_video_frames: The number of video frames each resulting sample will
      contain.
    video_step: The step between the video frames in the resulting sequence.
    initial_resize: When decoding the video, the frames will first be resized so
      that their shortest edge has this size.
    center_crop_size: When decoding the videos, the videos are first resized by
      `initial_resize`. We then split the video horizontally into
      `num_spatial_crops` crops, each crop having width `center_crop_size`. The
      height for each of the crops is always smpapled from the center of the
      (resized) video to a matching size of `center_crop_size`.

  Returns:
    A view containing videos of shape (N, T, H, W, 3), where N =
    `num_spatial_crops` X `num_temporal_crops`.
    T is `num_video_frames`, H = W = `center_crop_size`.

  """
  min_frames_required = (num_video_frames - 1) * video_step + 1
  sequence = media_sequences.extend_sequence(sequence, min_frames_required)

  sequences = _extract_temporal_crops(sequence, num_temporal_crops,
                                      num_video_frames, video_step)

  videos = []
  for subsequence in sequences:
    video = tf.map_fn(
        tf.io.decode_jpeg,
        subsequence.jpeg_encoded_images,
        fn_output_signature=tf.uint8)

    video = video_sampling.resize_min(video, initial_resize)
    video_height = tf.shape(video)[-3]
    video_width = tf.shape(video)[-2]

    horizontal_indices = sampling.compute_linearly_spaced_sample_indices(
        video_width, num_spatial_crops, center_crop_size, step=1)

    vertical_indices = sampling.compute_linearly_spaced_sample_indices(
        video_height, num_spatial_crops, center_crop_size, step=1)

    for vidx, hidx in zip(vertical_indices, horizontal_indices):
      v_start = vidx.start_index
      v_end = vidx.start_index + center_crop_size
      h_start = hidx.start_index
      h_end = hidx.start_index + center_crop_size

      video_sample = video[:, v_start:v_end, h_start:h_end, :]
      video_sample = tf.cast(video_sample, tf.float32)
      videos.append(video_sample)

  return brave_datasets.View(
      video=tf.stack(videos, axis=0),
      audio=None,
      labels=tf.tile(sequence.labels[tf.newaxis], (len(videos), 1)))


def _extract_temporal_crops(
    sequence, num_temporal_crops, num_video_frames,
    video_frame_step) -> Sequence[media_sequences.EncodedSequence]:

  sequence_length = tf.shape(sequence.jpeg_encoded_images)[0]
  temporal_crop_indices = sampling.compute_linearly_spaced_sample_indices(
      sequence_length, num_temporal_crops, num_video_frames, video_frame_step)

  return [
      time_sampling.get_subsequence_by_video_indices(sequence, indices)
      for indices in temporal_crop_indices
  ]
