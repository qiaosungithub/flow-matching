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
  model.image_size = 32
  model.out_channels = 1

  model.base_width = 64
  model.n_T = 18  # inference stepss
  model.dropout = 0.0

  model.use_aug_label = False
  model.average_loss = False

  model.sampler = 'euler' # or 'heun'
  model.ode_solver = 'jax'  # or 'scipy', which use RK45 solver
  model.net_type = 'ncsnpp'

  # # DDIM
  # model.beta_schedule = 'linear'
  # model.beta_start = 1e-4
  # model.beta_end = 0.02
  # model.num_diffusion_timesteps = 1000
  model.embedding_type = 'positional'

  config.aug = aug = ml_collections.ConfigDict()
  aug.use_edm_aug = False

  # Consistency training
  config.ct = ct = ml_collections.ConfigDict()
  ct.start_ema = 0.9
  ct.start_scales = 2
  ct.end_scales = 150

  # Dataset
  config.dataset = dataset = ml_collections.ConfigDict()
  dataset.name = 'MNIST'
  dataset.root = '/kmh-nfs-ssd-eu-mount/code/qiao/data/MNIST/'
  dataset.num_workers = 4
  dataset.prefetch_factor = 2
  dataset.pin_memory = False
  dataset.cache = False
  dataset.fake_data = False
  dataset.out_channels = 0  # from model
  dataset.steps_per_epoch = -1

  # Eval fid
  config.fid = fid = ml_collections.ConfigDict()
  fid.num_samples = 50000
  fid.fid_per_epoch = 500
  fid.on_use = True
  fid.eval_only = False
  fid.device_batch_size = 128
  # fid.cache_ref = '/kmh-nfs-us-mount/data/cached/cifar10_jax_stats_20240820.npz' # pytorch
  fid.cache_ref = '/kmh-nfs-us-mount/staging/zhh/data/cached/zhh_tfds_train_cifar10_stats_20241124.npz'

  # Training
  config.optimizer = 'sgd'

  config.learning_rate = 0.1
  config.lr_schedule = 'cosine'  # 'cosine'/'cos', 'const'

  config.weight_decay = 0.0001  
  config.adam_b1 = 0.9
  config.adam_b2 = 0.95
  config.grad_clip = 0.0

  config.warmup_epochs = 5.
  config.momentum = 0.9
  config.batch_size = 1024
  config.shuffle_buffer_size = 16 * 1024
  config.prefetch = 10

  config.num_epochs = 100
  config.log_per_step = 100
  config.log_per_epoch = -1
  config.eval_per_epoch = 1000
  config.visualize_per_epoch = 1
  config.checkpoint_per_epoch = 200

  config.steps_per_eval = -1

  config.restore = ''
  config.pretrain = ''

  config.half_precision = False

  config.seed = 0  # init random seed

  config.wandb = True
  config.load_from = None

  # evalu
  evalu = config.evalu = ml_collections.ConfigDict()
  evalu.ema = True
  evalu.sample = False # sample before testing fid

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
