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

#!/usr/bin/env python
"""A package to implement the BraVe model.

Broaden Your Views for Self-Supervised Video Learning.
"""

import setuptools

setuptools.setup(
    name='brave',
    python_requires='>3.7.0',
    version='0.0.1',
    description='Broaden Your Views for Self-Supervised Video Learning',
    author='Ross Hemsley',
    url='https://arxiv.org/abs/2103.16559',
    packages=setuptools.find_packages(),
    install_requires=[
        'absl-py==0.12.0',
        'astunparse~=1.6.3',
        'attrs~=21.2.0',
        'cached-property~=1.5.2',
        'cachetools~=4.2.2',
        'certifi~=2021.5.30',
        'charset-normalizer~=2.0.4',
        'chex==0.0.8',
        'contextlib2~=21.6.0',
        'dill==0.3.4',
        'dm-haiku==0.0.4',
        'dm-tree==0.1.6',
        'ffmpeg-python==0.2.0',
        'flatbuffers~=1.12',
        'future==0.18.2',
        'gast==0.4.0',
        'google-auth~=1.34.0',
        'google-auth-oauthlib==0.4.5',
        'google-pasta==0.2.0',
        'googleapis-common-protos~=1.53.0',
        'grpcio~=1.34.1',
        'h5py~=3.1.0',
        'idna~=3.2',
        'importlib-metadata~=4.6.3',
        'importlib-resources~=5.2.2',
        'iniconfig~=1.1.1',
        'jax==0.2.17',
        'jaxlib==0.1.69',
        'jaxline==0.0.3',
        'joblib~=1.0.1',
        'keras-nightly~=2.5.0.dev2021032900',
        'Keras-Preprocessing~=1.1.2',
        'Markdown~=3.3.4',
        'ml-collections==0.1.0',
        'mock~=4.0.3',
        'numpy~=1.19.5',
        'oauthlib~=3.1.1',
        'opt-einsum~=3.3.0',
        'optax==0.0.9',
        'packaging~=21.0',
        'pluggy==0.13.1',
        'promise~=2.3',
        'protobuf~=3.17.3',
        'py~=1.10.0',
        'pyasn1==0.4.8',
        'pyasn1-modules==0.2.8',
        'pyparsing~=2.4.7',
        'pytest~=6.2.4',
        'PyYAML~=5.4.1',
        'requests~=2.26.0',
        'requests-oauthlib~=1.3.0',
        'rsa~=4.7.2',
        'scikit-learn==1.0.1',
        'scipy~=1.5.4',
        'six~=1.15.0',
        'sklearn==0.0',
        'tabulate==0.8.9',
        'tensorboard~=2.5.0',
        'tensorboard-data-server==0.6.1',
        'tensorboard-plugin-wit~=1.8.0',
        'tensorflow~=2.5.0',
        'tensorflow-datasets~=4.4.0',
        'tensorflow-estimator~=2.5.0',
        'tensorflow-metadata~=1.2.0',
        'termcolor~=1.1.0',
        'threadpoolctl~=2.2.0',
        'toml==0.10.2',
        'toolz==0.11.1',
        'tqdm~=4.62.0',
        'typing-extensions~=3.7.4.3',
        'urllib3~=1.26.6',
        'Werkzeug~=2.0.1',
        'wrapt~=1.12.1',
        'zipp~=3.5.0',
    ])
