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

from functools import partial
import time
from typing import Any

from absl import logging
from flax import jax_utils as ju
from flax.training import common_utils
from flax.training.train_state import TrainState as FlaxTrainState
from flax.training import checkpoints
import orbax.checkpoint as ocp
import jax, os, wandb
from jax import lax, random
import jax.numpy as jnp
import ml_collections
import optax
import torch
import numpy as np
import flax.nnx as nn
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from torch.utils.data import DataLoader
from math import sqrt

from utils.info_util import print_params
from utils.vis_util import make_grid_visualization, visualize_cifar_batch
from utils.logging_util import log_for_0, Timer
from utils.metric_utils import tang_reduce
from utils.display_utils import show_dict, display_model, count_params
import utils.fid_util as fid_util
import utils.sample_util as sample_util

import models.models_ddpm as models_ddpm
from models.models_ddpm import generate, edm_ema_scales_schedules

NUM_CLASSES = 10

def get_input_pipeline(dataset_config):
    if dataset_config.name == 'imagenet2012:5.*.*':
        import input_pipeline_imgnet as input_pipeline
        return input_pipeline
    elif dataset_config.name == 'cifar10':
        import input_pipeline_cifar as input_pipeline
        return input_pipeline
    elif dataset_config.name == 'mnist':
        import input_pipeline_mnist as input_pipeline
        return input_pipeline
    else:
        raise ValueError('Unknown dataset {}'.format(dataset_config.name))

def compute_metrics(dict_losses):
  metrics = dict_losses.copy()
  metrics = lax.all_gather(metrics, axis_name='batch')
  metrics = jax.tree_map(lambda x: x.flatten(), metrics)  # (batch_size,)
  return metrics

def constant_lr_fn(base_learning_rate):
  return optax.constant_schedule(base_learning_rate)

def poly_decay_lr_fn(base_learning_rate, warmup_steps, total_steps):
  warmup_fn = optax.linear_schedule(
    init_value=1e-8,
    end_value=base_learning_rate,
    transition_steps=warmup_steps,
  )
  decay_fn = optax.polynomial_schedule(init_value=base_learning_rate, end_value=1e-8, power=1, transition_steps=total_steps-warmup_steps)
  return optax.join_schedules([warmup_fn, decay_fn], boundaries=[warmup_steps])

def create_learning_rate_fn(
  config: ml_collections.ConfigDict,
  base_learning_rate: float,
  steps_per_epoch: int,
):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
    init_value=0.0,
    end_value=base_learning_rate,
    transition_steps=config.warmup_epochs * steps_per_epoch,
  )
  if config.lr_schedule == 'poly':
    sched_fn = poly_decay_lr_fn(base_learning_rate, config.warmup_steps, config.num_epochs * steps_per_epoch)
  elif config.lr_schedule in ['constant', 'const']:
    sched_fn = constant_lr_fn(base_learning_rate)
  elif config.lr_schedule in ['cosine', 'cos']:
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    sched_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
  else:
    raise ValueError('Unknown learning rate scheduler {}'.format(config.lr_schedule))
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, sched_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch],
  )
  return schedule_fn

class NNXTrainState(FlaxTrainState):
  batch_stats: Any
  rng_states: Any
  graphdef: Any
  useless_variable_state: Any
  # NOTE: is_training can't be a attr, since it can't be replicated


def train_step_compute(state: NNXTrainState, batch, noise_batch, t_batch, learning_rate_fn, ema_scales_fn, config, data_scale=None):
  """
  Perform a single training step.
  We will pmap this function
  ---
  batch: a dict, with image, label, augment_label
  noise_batch: the noise_batch for the model
  t_batch: the t_batch for the model
  """

  ema_decay, scales = ema_scales_fn(state.step)

  def loss_fn(params_to_train):
    """loss function used for training."""
    
    outputs = state.apply_fn(state.graphdef, params_to_train, state.rng_states, state.batch_stats, state.useless_variable_state, True, batch['image'], batch['label'], batch['augment_label'], noise_batch, t_batch, data_scale=data_scale)
    loss, new_batch_stats, new_rng_states, dict_losses, images = outputs

    return loss, (new_batch_stats, new_rng_states, dict_losses, images)

  step = state.step
  dynamic_scale = None
  lr = learning_rate_fn(step)

  if dynamic_scale:
    raise NotImplementedError
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # aux, grads = grad_fn(state.params)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')

  # for simplicity, we don't all gather images
  # loss = aux[0]
  new_batch_stats, new_rng_states, dict_losses, images = aux[1]
  metrics = compute_metrics(dict_losses)
  metrics['lr'] = lr

  new_state = state.apply_gradients(
    grads=grads, batch_stats=new_batch_stats, rng_states=new_rng_states
  )

  # record ema
  metrics['ema_decay'] = ema_decay
  metrics['scales'] = scales
  # -------------------------------------------------------

  # -------------------------------------------------------
  # sanity
  # ema_outputs, _ = state.apply_fn(
  #     {'params': {'net': new_state.params['net_ema'],
  #                 'net_ema': new_state.params['net_ema'],},
  #      'batch_stats': state.batch_stats},
  #     batch['image'],
  #     batch['label'],
  #     mutable=['batch_stats'],
  #     rngs=dict(gen=rng_gen),
  # )
  # _, ema_dict_losses, _ = ema_outputs
  # ema_metrics = compute_metrics(ema_dict_losses)

  # metrics['ema_loss_train'] = ema_metrics['loss_train']
  # metrics['delta_loss_train'] = metrics['loss_train'] - ema_metrics['loss_train']
  # -------------------------------------------------------

  return new_state, metrics, images


def train_step(state: NNXTrainState, batch, rngs, train_step_compute_fn):
  """
  Perform a single training step.
  We will pmap this function
  ---
  batch: a dict, with image, label, augment_label
  rngs: nnx.Rngs
  train_step_compute_fn: the pmaped version of train_step_compute
  """

  # # ResNet has no dropout; but maintain rng_dropout for future usage
  # rng_step = random.fold_in(rng_init, state.step)
  # rng_device = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))
  # rng_gen, rng_dropout = random.split(rng_device)

  images = batch['image']
  # print("images.shape: ", images.shape) # (8, 64, 32, 32, 3)
  b1, b2 = images.shape[0], images.shape[1]
  noise_batch = jax.random.normal(rngs.train(), images.shape)
  t_batch = jax.random.uniform(rngs.train(), (b1, b2))

  new_state, metrics, images = train_step_compute_fn(state, batch, noise_batch, t_batch)

  return new_state, metrics, images

def train_step_sqa(state: NNXTrainState, batch, rngs, train_step_compute_fn):
  """
  Perform a single training step.
  We will pmap this function
  ---
  batch: a dict, with image, label, augment_label
  rngs: nnx.Rngs
  train_step_compute_fn: the pmaped version of train_step_compute
  """

  # # ResNet has no dropout; but maintain rng_dropout for future usage
  # rng_step = random.fold_in(rng_init, state.step)
  # rng_device = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))
  # rng_gen, rng_dropout = random.split(rng_device)

  images = batch['image']
  # print("images.shape: ", images.shape) # (8, 64, 32, 32, 3)
  b1, b2 = images.shape[0], images.shape[1]
  noise_batch = jax.random.normal(rngs.train(), images.shape)
  t_batch = jax.random.uniform(rngs.train(), (b1, b2))

  data_scale = jax.random.uniform(rngs.train(), (b1, b2)) * 0.0 + 1.0 # debug
  # disturb = jax.random.uniform(rngs.train(), (b1, b2)) # only disturb half of the data
  # data_scale = jnp.where(disturb < 0.5, data_scale, jnp.ones_like(data_scale))

  new_state, metrics, images = train_step_compute_fn(state, batch, noise_batch, t_batch, data_scale=data_scale)

  return new_state, metrics, images


def sample_step(state, sample_idx, model, rng_init, device_batch_size, MEAN_RGB=None, STDDEV_RGB=None):
  """
  sample_idx: each random sampled image corrresponds to a seed
  rng_init: here we do not want nnx.Rngs
  """
  rng_sample = random.fold_in(rng_init, sample_idx)  # fold in sample_idx
  images = generate(state, model, rng_sample, n_sample=device_batch_size)

  images_all = lax.all_gather(images, axis_name='batch')  # each device has a copy  
  images_all = images_all.reshape(-1, *images_all.shape[2:])

  # The images should be [-1, 1], which is correct

  # images_all = images_all * (jnp.array(STDDEV_RGB)/255.).reshape(1,1,1,3) + (jnp.array(MEAN_RGB)/255.).reshape(1,1,1,3)
  # images_all = (images_all - 0.5) / 0.5
  return images_all

def global_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  import random as R
  R.seed(seed)

def get_dtype(half_precision):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_dtype


checkpointer = ocp.StandardCheckpointer()
def _restore(ckpt_path, item, **restore_kwargs):
  return ocp.StandardCheckpointer.restore(checkpointer, ckpt_path, target=item)
setattr(checkpointer, 'restore', _restore)

def restore_checkpoint(model_init_fn, state, workdir, model_config, ema=False):
  # 杯子
  abstract_model = nn.eval_shape(lambda: model_init_fn(rngs=nn.Rngs(0), **model_config))
  rng_states = state.rng_states
  abs_state = nn.state(abstract_model)
  # params, batch_stats, others = abs_state.split(nn.Param, nn.BatchStat, ...)
  # useful_abs_state = nn.State.merge(params, batch_stats)
  _, useful_abs_state = abs_state.split(nn.RngState, ...)

  # abstract_model_1 = nn.eval_shape(lambda: model_init_fn(rngs=nn.Rngs(0), **model_config))
  # abs_state_1 = nn.state(abstract_model_1)
  # params_1, batch_stats_1, others = abs_state_1.split(nn.Param, nn.BatchStat, ...)
  # useful_abs_state_1 = nn.State.merge(params_1, batch_stats_1)

  fake_state = {
    'mo_xing': useful_abs_state,
    'ema_mo_xing': useful_abs_state,
    # 'ema_mo_xing': params_1,
    # 'ema_mo_xing': useful_abs_state_1,
    'you_hua_qi': state.opt_state,
    'step': 0
  }
  loaded_state = checkpoints.restore_checkpoint(workdir, target=fake_state,orbax_checkpointer=checkpointer)
  merged_params = loaded_state['mo_xing'] if not ema else loaded_state['ema_mo_xing']
  opt_state = loaded_state['you_hua_qi']
  step = loaded_state['step']
  params, batch_stats, _ = merged_params.split(nn.Param, nn.BatchStat, nn.VariableState)
  return state.replace(
    params=params,
    rng_states=rng_states,
    batch_stats=batch_stats,
    opt_state=opt_state,
    step=step
  )

# zhh's nnx version
def save_checkpoint(state:NNXTrainState, workdir, model_avg):
  state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
  model_avg = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model_avg))
  step = int(state.step)
  log_for_0('Saving checkpoint to {}, with step {}'.format(workdir, step))
  merged_params: nn.State = state.params
  # 不能把rng merge进去！
  # if len(state.rng_states) > 0:
  #     merged_params = nn.State.merge(merged_params, state.rng_states)
  if len(state.batch_stats) > 0:
    merged_params = nn.State.merge(merged_params, state.batch_stats)
  checkpoints.save_checkpoint_multiprocess(workdir, {
    'mo_xing': merged_params,
    'ema_mo_xing': model_avg,
    'you_hua_qi': state.opt_state,
    'step': step
  }, step, keep=2, orbax_checkpointer=checkpointer)
  # NOTE: this is tang, since "keep=2" means keeping the most recent 3 checkpoints.

# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state: NNXTrainState):
  """Sync the batch statistics across replicas. This is called before evaluation."""
  # Each device has its own version of the running average batch statistics and
  if hasattr(state, 'batch_stats'):
    return state
  if len(state.batch_stats) == 0:
    return state
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

# def get_no_weight_decay_dict(params):
#   def modify_value_based_on_key(obj):
#     if not isinstance(obj, dict):
#       return obj
#     for k,v in obj.items():
#       if not isinstance(v,dict):
#         if k in {'cls','pos_emb','bias','scale'}:
#           obj[k] = False
#         else:
#           obj[k] = True
#     return obj
#   def is_leaf(obj):
#     if not isinstance(obj, dict):
#       return True
#     modify_value_based_on_key(obj)
#     b = isinstance(obj, dict) and all([not isinstance(v, dict) for v in obj.values()])
#     return b
#   u = jax.tree_util.tree_map(lambda x:False,params)
#   modified_tree = jax.tree_util.tree_map(partial(modify_value_based_on_key), u, is_leaf=is_leaf)
#   return modified_tree

def create_train_state(
  config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
  """
  Create initial training state, including the model and optimizer.
  config: the training config
  """
  # print("here we are in the function 'create_train_state' in train.py; ready to define optimizer")
  graphdef, params, batch_stats, rng_states, useless_variable_states = nn.split(model, nn.Param, nn.BatchStat, nn.RngState, nn.VariableState)

  print_params(params)

  def apply_fn(graphdef2, params2, rng_states2, batch_stats2, useless_, is_training, images, labels, augment_labels, noise_batch, t_batch, data_scale=None):
    """
    input:
      images
      labels
      augment_labels: we condition our network on the augment_labels
    ---
    output:
      loss_train
      new_batch_stats
      new_rng_states
      dict_losses: contains loss and loss_train, which are the same
      images: all predictions and images and noises
    """
    merged_model = nn.merge(graphdef2, params2, rng_states2, batch_stats2, useless_)
    if is_training:
      merged_model.train()
    else:
      merged_model.eval()
    del params2, rng_states2, batch_stats2, useless_
    loss_train, dict_losses, images = merged_model.forward(images, labels, augment_labels, noise_batch, t_batch, data_scale=data_scale)
    new_batch_stats, new_rng_states, _ = nn.state(merged_model, nn.BatchStat, nn.RngState, ...)
    return loss_train, new_batch_stats, new_rng_states, dict_losses, images

  # here is the optimizer

  if config.optimizer == 'sgd':
    log_for_0('Using SGD')
    tx = optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=config.momentum,
      nesterov=True,
    )
  elif config.optimizer == 'adamw':
    log_for_0(f'Using AdamW with wd {config.weight_decay}')
    tx = optax.adamw(
      learning_rate=learning_rate_fn,
      b1=config.adam_b1,
      b2=config.adam_b2,
      weight_decay=config.weight_decay,
      # mask=mask_fn,  # TODO{km}
    )
  elif config.optimizer == 'radam':
    log_for_0(f'Using RAdam with wd {config.weight_decay}')
    assert config.weight_decay == 0.0
    tx = optax.radam(
      learning_rate=learning_rate_fn,
      b1=config.adam_b1,
      b2=config.adam_b2,
    )
  else:
    raise ValueError(f'Unknown optimizer: {config.optimizer}')
  
  state = NNXTrainState.create(
    graphdef=graphdef,
    apply_fn=apply_fn,
    params=params,
    tx=tx,
    batch_stats=batch_stats,
    useless_variable_state=useless_variable_states,
    rng_states=rng_states,
  )
  return state

def prepare_batch_data(batch, config, batch_size=None):
  """Reformat a input batch from TF Dataloader.
  
  Args:
    batch: dict
      image: shape (b1, b2, h, w, c)
      label: shape (b1, b2)
    batch_size = expected batch_size of this node, for eval's drop_last=False only
  """
  image, label = batch["image"], batch["label"]
  # print("In prepare_batch_data, image.shape: ", image.shape) # (8, 64, 32, 32, 3)
  # print("In prepare_batch_data, label.shape: ", label.shape) # (8, 64)

  if config.aug.use_edm_aug:
    raise NotImplementedError
    augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
    image, augment_label = augment_pipe(image)
  else:
    augment_label = None

  # pad the batch if smaller than batch_size
  if batch_size is not None and batch_size > image.shape[0]:
    raise ValueError("not supported")
    image = np.cat([image, np.zeros((batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype)], axis=0)
    label = np.cat([label, -np.ones((batch_size - label.shape[0],), dtype=label.dtype)], axis=0)
    assert augment_label is None  # don't support padding augment_label

  # reshape (host_batch_size, 3, height, width) to
  # (local_devices, device_batch_size, height, width, 3)
  local_device_count = jax.local_device_count()
  assert image.shape[0] == local_device_count

  if config.model.use_aug_label:
    assert config.aug.use_edm_aug
    augment_label = augment_label.reshape((local_device_count, -1) + augment_label.shape[1:])
    augment_label = augment_label.numpy()
  else:
    augment_label = None

  return_dict = {
    'image': image,
    'label': label,
    'augment_label': augment_label,
  }

  return return_dict

def _update_model_avg(model_avg, state_params, ema_decay):
  return jax.tree_util.tree_map(lambda x, y: ema_decay * x + (1.0 - ema_decay) * y, model_avg, state_params)

def train_and_evaluate(
  config: ml_collections.ConfigDict, workdir: str
) -> NNXTrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  ########### Initialize ###########
  rank = index = jax.process_index()
  config.dataset.out_channels = config.model.out_channels
  model_config = config.model 
  dataset_config = config.dataset
  if rank == 0 and config.wandb:
    # wandb.init(project='sqa_FM_kaiming_copied_nnx', dir=workdir)
    wandb.init(project='LMCI-eval', dir=workdir, tags=["SQA-pt"])
    # wandb.init(project='sqa_FM_compare', dir=workdir)
    wandb.config.update(config.to_dict())
  global_seed(config.seed)

  image_size = model_config.image_size

  log_for_0('config.batch_size: {}'.format(config.batch_size))

  rngs = nn.Rngs(config.seed)
  ########### Create DataLoaders ###########

  input_pipeline = get_input_pipeline(dataset_config)
  input_type = tf.bfloat16 if config.half_precision else tf.float32
  dataset_builder = tfds.builder(dataset_config.name)
  assert config.batch_size % jax.process_count() == 0, ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()
  assert local_batch_size % jax.local_device_count() == 0, ValueError('Local batch size must be divisible by the number of local devices')
  log_for_0('local_batch_size: {}'.format(local_batch_size))
  log_for_0('jax.local_device_count: {}'.format(jax.local_device_count()))
  log_for_0('global batch_size: {}'.format(config.batch_size))
  train_loader, steps_per_epoch, yierbayiyiliuqi = input_pipeline.create_split(
    dataset_builder,
    dataset_config=dataset_config,
    training_config=config,
    local_batch_size=local_batch_size,
    input_type=input_type,
    train=False if dataset_config.fake_data else True
  )
  val_loader, val_steps, _ = input_pipeline.create_split(
    dataset_builder,
    dataset_config=dataset_config,
    training_config=config,
    local_batch_size=local_batch_size,
    input_type=input_type,
    train=False
  )
  if dataset_config.fake_data:
    log_for_0('Note: using fake data')
  log_for_0('steps_per_epoch: {}'.format(steps_per_epoch))
  log_for_0('eval_steps: {}'.format(val_steps))

  # eval points
  time = [config.cal.eval_time]
  gap = config.cal.gap
  n_p = config.cal.num_points
  n_n = config.cal.nn # number of noisy images to eval
  sde = config.cal.sde

  assert n_p % 2 == 1

  ########### prepare noisy image ###########
  log_for_0('start preparing noisy images')
  for n_batch, batch in zip(range(steps_per_epoch), train_loader):
    batch = prepare_batch_data(batch, config)
    data = batch['image'].reshape(-1, 32, 32, 3)
    break
  noisy_images = []
  ind=0
  for t in time:
    data_this = data[ind:ind+n_n]
    ind += n_n
    noise = jax.random.normal(rngs(), data_this.shape)
    if sde == "flow": noisy = (1-t) * data_this + t * noise
    elif sde == "VP": noisy = sqrt(1-t) * data_this + sqrt(t) * noise
    else: raise NotImplementedError
    noisy_images.append(noisy)

  log_for_0('finish preparing noisy images')

  # # logging visualizations
  # vis = visualize_cifar_batch(vis)
  # # print("vis.shape: ", vis.shape) # (8, 160, 256, 3)
  # vis = jax.device_get(vis)
  # vis = vis[0]
  # canvas = Image.fromarray(vis)
  # if config.wandb and index == 0:
  #   wandb.log({'visualize': wandb.Image(canvas)})

  ########### calculate ###########
  for t in time:
    # p(t|z)=\sum p(t|z, x)p(x|z)=\sum p(eps=(z-(1-t)x)/t)p(x|z)
    eval_time = jnp.arange(n_p) * gap - gap * (n_p // 2) + t # eval time steps
    assert jnp.all(eval_time >= 1e-4) and jnp.all(eval_time <= 1-1e-4) # check
    eval_time = eval_time.reshape(-1, 1, 1, 1)
    log_coeff = jnp.log(eval_time) * (-32*32*3)
    log_coeff = log_coeff.reshape(n_p, 1, 1)
    total_log_prob = []
    noisy = noisy_images.pop(0)
    assert noisy.shape == (n_n, 32, 32, 3)
    noisy = noisy.reshape(1, 1, n_n, 32*32*3).repeat(n_p, axis=0) # (n_p, 1, n_n, 3096)
    log_for_0(f'start calculating for t={t}')
    for n_batch, batch in zip(range(steps_per_epoch), train_loader):
      batch = prepare_batch_data(batch, config)
      data = batch['image']
      data = data.reshape(-1, 32, 32, 3)
      b = data.shape[0]
      assert data.shape == (b, 32, 32, 3)
      data = data.reshape(1, b, 1, 32*32*3).repeat(n_p, axis=0) # (n_p, b, 1, 3096)
      # we hope to get shape (n_p, b, n_n, 3096)
      if sde == "flow": noise = (noisy - ((1 - eval_time) * data)) / eval_time # (n_p, b, n_n, 3096)
      elif sde == "VP": noise = (noisy - (sqrt(1 - eval_time) * data)) / sqrt(eval_time) # (n_p, b, n_n, 3096)
      else: raise NotImplementedError("我写了")
      # calculate the pr of noise
      norm = jnp.sum(noise ** 2, axis=-1) # (n_p, b, n_n)
      # print("norm: ", jnp.mean(norm))
      # cast to float64
      # norm = jnp.array(norm, dtype=jnp.float64)
      # norm = norm - 32 * 32 * 3 # remember to divide by exp(-1/2 * 32*32*3)
      log_prob = -0.5 * norm + log_coeff
      log_prob = jax.nn.logsumexp(log_prob, axis=1) # (n_p, n_n)
      # print("log_prob: ", log_prob)
      total_log_prob.append(log_prob)
    # gather total_log_prob to a (N, n_p, n_n) array
    total_log_prob = jnp.stack(total_log_prob, axis=0)
    total_log_prob = jax.nn.logsumexp(total_log_prob, axis=0) # (n_p, n_n)

    # sum = sum / 50000 # remember to divide by 50000
    # log
    assert config.wandb
    eval_time = eval_time.flatten()
    if config.wandb and index == 0:
      for i in range(n_p):
        et = eval_time[i]
        dic = {}
        dic['eval_time'] = et
        for j in range(n_n):
          dic[f'tlp{j}'] = total_log_prob[i, j]
        wandb.log(dic)


    for j in range(n_n):
      log_for_0(f'results for n={j}')
      tlp = total_log_prob[:, j]
      for i in range(n_p):
        s = tlp[i]
        log_for_0(f'eval_time: {eval_time[i]}, mlp: {s}')
        # if config.wandb and index == 0:
        #   wandb.log({
        #     f'mlp{j}': s,
        #     })
    # if config.wandb and index == 0:
    #   for i in range(n_p):
    #     et = eval_time[i]
    #     wandb.log({
    #       'eval_time': et,
    #       })
      

  # # log the line plot
  # if config.wandb and index == 0:
  #   wandb.log({
  #       "eval_time_vs_sum": wandb.plot.line_series(
  #           xs=eval_time.flatten(),
  #           ys=sum.flatten(),
  #           keys=["sum"],
  #           title="Sum vs Eval Time",
  #           xname="Eval Time"
  #       )
  #   })

  return 0

def just_evaluate(
    config: ml_collections.ConfigDict, workdir: str
  ):
  # assert the version of orbax-checkpoint is 0.4.4
  assert ocp.__version__ == '0.6.4', ValueError(f'orbax-checkpoint version must be 0.6.4, but got {ocp.__version__}')
  ########### Initialize ###########
  rank = index = jax.process_index()
  model_config = config.model 
  dataset_config = config.dataset
  fid_config = config.fid
  if rank == 0 and config.wandb:
    wandb.init(project='sqa_FM_nnx_evaluate', dir=workdir)
    # wandb.init(project='sqa_edm_debug', dir=workdir)
    wandb.config.update(config.to_dict())
  # dtype = jnp.bfloat16 if model_config.half_precision else jnp.float32
  global_seed(config.seed)
  image_size = model_config.image_size

  if rank == 0 and config.wandb:
    wandb.log({'n_T': config.model.n_T})

  ########### Create Model ###########
  model_cls = models_ddpm.SimDDPM
  rngs = nn.Rngs(config.seed, params=config.seed + 114, dropout=config.seed + 514, train=config.seed + 1919)
  dtype = get_dtype(config.half_precision)
  model_init_fn = partial(model_cls, num_classes=NUM_CLASSES, dtype=dtype)
  model = model_init_fn(rngs=rngs, **model_config)
  show_dict(f'number of model parameters:{count_params(model)}')
  # show_dict(display_model(model))

  ########### Create LR FN ###########
  learning_rate_fn = lambda:1 # just in order to create the state

  ########### Create Train State ###########
  state = create_train_state(config, model, image_size, learning_rate_fn)
  assert config.get('load_from',None) is not None, 'Must provide a checkpoint path for evaluation'
  if not os.path.isabs(config.load_from):
    raise ValueError('Checkpoint path must be absolute')
  if not os.path.exists(config.load_from):
    raise ValueError('Checkpoint path {} does not exist'.format(config.load_from))
  state = restore_checkpoint(model_init_fn, state, config.load_from, model_config, ema=config.evalu.ema) # NOTE: whether to use the ema model
  state_step = int(state.step)
  state = ju.replicate(state) # NOTE: this doesn't split the RNGs automatically, but it is an intended behavior

  
  # ### debug edm here
  # # state = state[0]
  # t = model.compute_t(jnp.arange(35), 35)
  # vis, denoised = generate(state, model, random.PRNGKey(0), 1) # (num_steps, 32, 32, 3)
  # print("vis.shape: ", vis.shape)
  # vis = vis.reshape(35, 32, 32, 3)
  # denoised = denoised.reshape(35, 32, 32, 3)
  # assert vis.shape == (35, 32, 32, 3)
  # assert denoised.shape == (35, 32, 32, 3)
  # from utils.vis_util import float_to_uint8
  # for ep in range(35):
  #   img = vis[ep]
  #   max = np.max(img)
  #   min = np.min(img)
  #   mean = np.sqrt(np.mean(img**2))
  #   img = float_to_uint8(img)
  #   denoised_img = denoised[ep]
  #   denoised_img = float_to_uint8(denoised_img)
  #   if index == 0 and config.wandb:
  #     wandb.log({
  #       'ep': ep,
  #       'max': max,
  #       'min': min,
  #       'mean': mean,
  #       'img': wandb.Image(img),
  #       'denoised': wandb.Image(denoised_img),
  #       'noise_level': t[ep]
  #       })
    

  # exit("6.7900")

  ########### FID ###########
  vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())  # for visualization
  if config.model.ode_solver == 'jax':
    p_sample_step = jax.pmap(
      partial(sample_step, 
              model=model, 
              rng_init=random.PRNGKey(0), 
              device_batch_size=config.fid.device_batch_size, 
              # MEAN_RGB=input_pipeline.MEAN_RGB, 
              # STDDEV_RGB=input_pipeline.STDDEV_RGB
      ),
      axis_name='batch'
    )

    def run_p_sample_step(p_sample_step, state, sample_idx):
      """
      state: train state
      """
      # redefine the interface
      images = p_sample_step(state, sample_idx=sample_idx)
      # print("In function run_p_sample_step; images.shape: ", images.shape, flush=True)
      jax.random.normal(random.key(0), ()).block_until_ready()
      return images[0]  # images have been all gathered
    
  elif config.model.ode_solver == 'scipy':
    from utils.rk45_util import get_rk45_functions
    run_p_sample_step, p_sample_step = get_rk45_functions(model, config, random.PRNGKey(0))

  else:
    raise NotImplementedError
  # ------------------------------------------------------------------------------------
  if config.fid.on_use:  # we will evaluate fid    
    inception_net = fid_util.build_jax_inception()
    stats_ref = fid_util.get_reference(config.fid.cache_ref, inception_net)

    # if config.fid.eval_only: # debug, this is tang
    #   samples_all = sample_util.generate_samples_for_fid_eval(state, workdir, config, p_sample_step, run_p_sample_step)
    #   mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
    #   fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
    #   log_for_0(f'w/o ema: FID at {samples_all.shape[0]} samples: {fid_score}')

    #   samples_all = sample_util.generate_samples_for_fid_eval(state, workdir, config, p_sample_step, run_p_sample_step)
    #   mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
    #   fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
    #   log_for_0(f' w/ ema: FID at {samples_all.shape[0]} samples: {fid_score}')
    #   return None

    # debugging here
    # samples_dir = '/kmh-nfs-us-mount/logs/kaiminghe/results-edm/edm-cifar10-32x32-uncond-vp'
    # samples = sample_util.get_samples_from_dir(samples_dir, config)
  # ------------------------------------------------------------------------------------

  ########### Gen ###########

  log_for_0(f'fixed_sample_idx: {vis_sample_idx}')
  log_for_0('Eval...')
  ########### Sampling ###########
  eval_state = sync_batch_stats(state)
  if config.evalu.sample: # if we want to sample
    log_for_0(f'Sample...')
    # sync batch statistics across replicas
    # eval_state = eval_state.replace(params=model_avg)
    vis = run_p_sample_step(p_sample_step, eval_state, vis_sample_idx)
    vis = make_grid_visualization(vis)
    vis = jax.device_get(vis) # np.ndarray
    vis = vis[0]
    # print(vis.shape)
    # exit("王广廷")
    canvas = Image.fromarray(vis)
    if config.wandb and index == 0:
      wandb.log({'gen': wandb.Image(canvas)})
    # sample_step(eval_state, image_size, sampling_config, epoch, use_wandb=config.wandb)
  ########### FID ###########
  if config.fid.on_use:

    samples_all = sample_util.generate_samples_for_fid_eval(eval_state, workdir, config, p_sample_step, run_p_sample_step)
    mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
    fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
    log_for_0(f'FID at {samples_all.shape[0]} samples: {fid_score}')

    if config.wandb and rank == 0:
      wandb.log({
        'FID': fid_score,
      })

    vis = make_grid_visualization(samples_all, to_uint8=False)
    vis = jax.device_get(vis)
    vis = vis[0]
    canvas = Image.fromarray(vis)
    if config.wandb and index == 0:
      wandb.log({'gen_fid': wandb.Image(canvas)})

  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  if index == 0 and config.wandb:
    wandb.finish()