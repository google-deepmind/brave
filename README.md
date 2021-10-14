# BraVe

This is a [JAX](https://jax.readthedocs.io/) implementation of
[*Broaden Your Views for Self-Supervised Video Learning*](https://arxiv.org/abs/2103.16559),
or *BraVe* for short.

The model provided in this package was implemented based on the internal model
that was used to compute results for the accompanying paper. It achieves
comparable results on the evaluation tasks when evaluated side-by-side. Not all
details are guaranteed to be identical though, and some results may differ from
those given in the paper. In particular, this implementation does not provide
the option to train with optical flow.

We provide a selection of pretrained checkpoints in the table below, which can
directly be evaluated against HMDB 51 with the evaluation tools this package.
These are exactly the checkpoints that were used to provide the numbers in the
accompanying paper, and were not trained with the exact trainer given in this
package. For details on training a model with this package, please see the end
of this readme.

In the table below, the different configurations are represented by using e.g.
V/A for video (narrow view) to audio (broad view), or V/F for a narrow view
containing video, and a broad view containing optical flow.

The backbone in each case is TSMResnet, with a given width multiplier (please
see the accompanying paper for further details). For all of the given numbers
below, the SVM regularization constant used is 0.0001. For HMDB 51, the average
is given in brackets, followed by the top-1 percentages for each of the splits.

<!-- mdformat off(allow long lines) -->
Views         | Architecture  | HMDB51                             | UCF-101 | K600  | Trained with this package   | Checkpoint
------------- |:-------------:|:----------------------------------:|:-------:|:-----:|:---------------------------:| -----------
V/AF          | TSM (1X)      | (69.2%) 71.307%, 68.497%, 67.843%  | 92.9%   | 69.2% | ✗                           | [download](https://storage.googleapis.com/dm-jaxline/brave/29557527_2_0.npy)
V/AF          | TSM (2X)      | (69.9%) 72.157%, 68.432%, 69.02%   | 93.2%   | 70.2% | ✗                           | [download](https://storage.googleapis.com/dm-jaxline/brave/29509415_1_0.npy)
V/A           | TSM (1X)      | (69.4%) 70.131%, 68.889%, 69.085%  | 93.0%   | 70.6% | ✗                           | [download](https://storage.googleapis.com/dm-jaxline/brave/29466668_1_0.npy)
V/VVV         | TSM (1X)      | (65.4%) 66.797%, 63.856%, 65.425%  | 92.6%   | 70.8% | ✗                           | [download](https://storage.googleapis.com/dm-jaxline/brave/29332798_5_0.npy)
<!-- mdformat on -->

## Reproducing results from the paper

This package provides everything needed to evaluate the above checkpoints
against HMDB 51. It supports Python 3.7 and above.

To get started, we recommend using a clean virtualenv. You may then install the
brave package directly from GitHub using,

```bash
pip install git+https://github.com/deepmind/brave.git
```

A pre-processed version of the HMDB 51 dataset can be downloaded using the
following command. It requires that both `ffmpeg` and `unrar` are available. The
following will download the dataset to `/tmp/hmdb51/`, but any other location
would also work.

```bash
  python -m brave.download_hmdb --output_dir /tmp/hmdb51/
```

To evaluate a checkpoint downloaded from the above table, the following may be
used. The dataset shards arguments should be set to match the paths used above.

```bash
  python -m brave.evaluate_video_embeddings \
    --checkpoint_path <path/to/downloaded/checkpoint>.npy \
    --train_dataset_shards '/tmp/hmdb51/split_1/train/*' \
    --test_dataset_shards '/tmp/hmdb51/split_1/test/*' \
    --svm_regularization 0.0001 \
    --batch_size 8
```

Note that any of the three splits can be evaluated by changing the dataset split
paths. To run this efficiently using a GPU, it is also necessary to install the
correct version of `jaxlib`. To install jaxlib with support for cuda 10.1 on
linux, the following install should be sufficient, though other precompiled
packages may be found through the JAX documentation.

```bash
  pip install https://storage.googleapis.com/jax-releases/cuda101/jaxlib-0.1.69+cuda101-cp39-none-manylinux2010_x86_64.whl
```

Depending on the available GPU memory available, the `batch_size` parameter may
be tuned to obtain better performance, or to reduce the required GPU memory.

## Training a network

This package may also be used to train a model from scratch using
[jaxline](https://github.com/deepmind/jaxline). In order to try this, first
ensure the configuration is set appropriately by modifying `brave/config.py`. At
minimum, it would also be necessary to choose an appropriate global batch size
(by default, the setting of 512 is likely too large for any single-machine
training setup). In addition, a value must be set for `dataset_shards`. This
should contain the paths of the tfrecord files containing the serialized
training data.

For details on checkpointing and distributing computation, see the
[jaxline documentation](https://github.com/deepmind/jaxline).

Similarly to above, it is necessary to install the correct `jaxlib` package to
enable training on a GPU.

The training may now be launched using,

```bash
  python -m brave.experiment --config=brave/config.py
```

### Training datasets

This model is able to read data stored in the format specified by
[DMVR](https://github.com/deepmind/dmvr). For an example of writing training
data in the correct format see the code in `dataset/fixtures.py`, which is used
to write the test fixtures used in the tests for this package.

## Running the tests

After checking out this code locally, you may run the package tests using

```bash
  pip install -e .
  pytest brave
```

We recommend doing this from a clean virtual environment.

## Citing this work

If you use this code (or any derived code), data or these models in your work,
please cite the relevant accompanying [paper](https://arxiv.org/abs/2103.16559).

```
@misc{recasens2021broaden,
      title={Broaden Your Views for Self-Supervised Video Learning},
      author={Adrià Recasens and Pauline Luc and Jean-Baptiste Alayrac and Luyu Wang and Ross Hemsley and Florian Strub and Corentin Tallec and Mateusz Malinowski and Viorica Patraucean and Florent Altché and Michal Valko and Jean-Bastien Grill and Aäron van den Oord and Andrew Zisserman},
      year={2021},
      eprint={2103.16559},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Disclaimer

This is not an official Google product
