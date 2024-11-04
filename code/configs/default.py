# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""

  ################ WARNING ################
  # DO NOT DIRECTLY MODIFY THIS FILE, IN  #
  # ANY WAY. USE EXP_CONFIG.YML TO SET    #
  # INSTEAD.                              #
  #########################################

  config = ml_collections.ConfigDict()

  # Model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'NCSNv2'
  model.half_precision = False
  model.spec_norm = False
  model.normalization = "InstanceNorm++"
  model.activation = "elu"
  model.ngf = 64

  # Dataset
  config.dataset = dataset = ml_collections.ConfigDict()
  dataset.name = 'MNIST'
  dataset.image_size = 28
  dataset.channels = 1
  dataset.root = '/kmh-nfs-ssd-eu-mount/code/qiao/data/MNIST/'
  dataset.num_workers = 4
  dataset.prefetch_factor = 2
  dataset.pin_memory = False
  dataset.cache = True

  # Training
  config.training = training = ml_collections.ConfigDict()
  training.learning_rate = 0.1
  # config.momentum = 0.9
  training.batch_size = 128
  training.eval_batch_size = 500
  training.shuffle_buffer_size = 16 * 128
  # config.prefetch = 10
  training.weight_decay = 0.0 

  training.num_epochs = 100
  training.wandb = True
  training.log_per_step = 100
  training.log_per_epoch = -1
  training.eval_per_epoch = 1
  training.checkpoint_per_epoch = 20
  training.checkpoint_max_keep = 2
  training.steps_per_eval = -1
  training.seed = 3407  # init random seed
  training.load_from = None

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.sigma_begin = 28
  sampling.n_noise_levels = 75
  sampling.sigma_end = 0.01
  sampling.ema = True
  sampling.ema_decay = 0.999
  sampling.eps = 5e-5
  sampling.T = 5
  sampling.save_dir = '/kmh-nfs-ssd-eu-mount/code/qiao/NCSN/sqa_NCSN/images/'

  ################ WARNING ################
  # DO NOT DIRECTLY MODIFY THIS FILE, IN  #
  # ANY WAY. USE EXP_CONFIG.YML TO SET    #
  # INSTEAD.                              #
  #########################################

  return config


def metrics():
  return [
    'train_loss',
    'eval_loss',
    'train_accuracy',
    'eval_accuracy',
    'steps_per_second',
    'train_learning_rate',
  ]
