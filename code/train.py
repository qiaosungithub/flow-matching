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

from utils.info_util import print_params
from utils.vis_util import make_grid_visualization, visualize_cifar_batch
from utils.logging_util import log_for_0, Timer
from utils.metric_utils import tang_reduce
from utils.display_utils import show_dict, display_model, count_params
import utils.fid_util as fid_util
import utils.sample_util as sample_util

import models.models_ddpm as models_ddpm
from models.models_ddpm import generate, edm_ema_scales_schedules
import input_pipeline
from input_pipeline import prepare_batch_data

NUM_CLASSES = 10


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


def train_step_compute(state: NNXTrainState, batch, noise_batch, t_batch, learning_rate_fn, ema_scales_fn, config):
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
    
    outputs = state.apply_fn(state.graphdef, params_to_train, state.rng_states, state.batch_stats, state.useless_variable_state, True, batch['image'], batch['label'] if config.model.class_conditional else None, batch['augment_label'], noise_batch, t_batch)
    loss, new_batch_stats, new_rng_states, dict_losses, images = outputs

    return loss, (new_batch_stats, new_rng_states, dict_losses, images)

  step = state.step
  dynamic_scale = None
  lr = learning_rate_fn(step)

  if dynamic_scale:
    raise NotImplementedError('dynamic_scale is not implemented')
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # aux, grads = grad_fn(state.params)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')

  # # clip grad with config.grad_clip
  # if config.grad_clip > 0.0:
  #   grads = optax.clip_by_global_norm(grads, config.grad_clip)[1]
  # TODO: clip grad

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


def train_step(state: NNXTrainState, batch, rngs, train_step_compute_fn, config):
  """
  Perform a single training step.
  This function is NOT pmap, so you can do anything
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
  if config.model.task == 'FM':
    t_batch = jax.random.uniform(rngs.train(), (b1, b2))
  elif config.model.task == 'Diffusion':
    t_batch = jax.random.randint(rngs.train(), (b1, b2), minval=0, maxval=config.diffusion.diffusion_nT) # [0, num_time_steps)
  else:
    raise NotImplementedError('Unknown task: {}'.format(config.model.task))

  new_state, metrics, images = train_step_compute_fn(state, batch, noise_batch, t_batch)

  return new_state, metrics, images


def sample_step(state, sample_idx, model, rng_init, device_batch_size, config,MEAN_RGB=None, STDDEV_RGB=None,option='FID'):
  """
  sample_idx: each random sampled image corrresponds to a seed
  rng_init: here we do not want nnx.Rngs
  """
  rng_sample = random.fold_in(rng_init, sample_idx)  # fold in sample_idx
  images, nfe = generate(state, model, rng_sample, n_sample=device_batch_size,config=config,label_type=('order' if option=='vis' else 'random') if config.model.class_conditional else 'none')
  assert nfe is not None, 'Returning None as NFE is deprecated'
  nfe = jnp.array(nfe)

  images_all = lax.all_gather(images, axis_name='batch')  # each device has a copy  
  images_all = images_all.reshape(-1, *images_all.shape[2:])

  # The images should be [-1, 1], which is correct

  # images_all = images_all * (jnp.array(STDDEV_RGB)/255.).reshape(1,1,1,3) + (jnp.array(MEAN_RGB)/255.).reshape(1,1,1,3)
  # images_all = (images_all - 0.5) / 0.5
  return images_all, jax.device_get(nfe).mean()

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
  ), loaded_state['ema_mo_xing'] if not ema else loaded_state['mo_xing']

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

  def apply_fn(graphdef2, params2, rng_states2, batch_stats2, useless_, is_training, images, labels, augment_labels, noise_batch, t_batch):
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
    loss_train, dict_losses, images = merged_model.forward(images, labels, augment_labels, noise_batch, t_batch)
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
  # elif config.optimizer == 'adam':
  #   log_for_0(f'Using Adam')
  #   tx = optax.adam(
  #     learning_rate=learning_rate_fn,
  #     b1=config.adam_b1,
  #     b2=config.adam_b2,
  #   )
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

# def prepare_batch_data(batch, config, batch_size=None):
#   """Reformat a input batch from TF Dataloader.
  
#   Args:
#     batch: dict
#       image: shape (b1, b2, h, w, c)
#       label: shape (b1, b2)
#     batch_size = expected batch_size of this node, for eval's drop_last=False only
#   """
#   image, label = batch["image"], batch["label"]
#   # print("In prepare_batch_data, image.shape: ", image.shape) # (8, 64, 32, 32, 3)
#   # print("In prepare_batch_data, label.shape: ", label.shape) # (8, 64)

#   if config.aug.use_edm_aug:
#     raise NotImplementedError
#     augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
#     image, augment_label = augment_pipe(image)
#   else:
#     augment_label = None

#   # pad the batch if smaller than batch_size
#   if batch_size is not None and batch_size > image.shape[0]:
#     raise ValueError("not supported")
#     image = np.cat([image, np.zeros((batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype)], axis=0)
#     label = np.cat([label, -np.ones((batch_size - label.shape[0],), dtype=label.dtype)], axis=0)
#     assert augment_label is None  # don't support padding augment_label

#   # reshape (host_batch_size, 3, height, width) to
#   # (local_devices, device_batch_size, height, width, 3)
#   local_device_count = jax.local_device_count()
#   assert image.shape[0] == local_device_count

#   if config.model.use_aug_label:
#     assert config.aug.use_edm_aug
#     augment_label = augment_label.reshape((local_device_count, -1) + augment_label.shape[1:])
#     augment_label = augment_label.numpy()
#   else:
#     augment_label = None

#   return_dict = {
#     'image': image,
#     'label': label,
#     'augment_label': augment_label,
#   }

#   return return_dict

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
  fid_config = config.fid
  if rank == 0 and config.wandb:
    wandb.init(project='LMCI', dir=workdir, tags=['Sanity_Check'])
    # wandb.init(project='sqa_FM_compare', dir=workdir)
    wandb.config.update(config.to_dict())
  global_seed(config.seed)

  image_size = model_config.image_size

  log_for_0('config.batch_size: {}'.format(config.batch_size))

  # # print("save dir: ", sampling_config.save_dir)
  # if sampling_config.save_dir is None:
  #   sampling_config.save_dir = workdir + "/images/"
  # log_for_0(f"save directory: {sampling_config.save_dir}")

  ########### Create DataLoaders ###########


  # input_pipeline = get_input_pipeline(dataset_config)
  # input_type = tf.bfloat16 if config.half_precision else tf.float32
  # dataset_builder = tfds.builder(dataset_config.name)
  assert config.batch_size % jax.process_count() == 0, ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()
  assert local_batch_size % jax.local_device_count() == 0, ValueError('Local batch size must be divisible by the number of local devices')
  log_for_0('local_batch_size: {}'.format(local_batch_size))
  log_for_0('jax.local_device_count: {}'.format(jax.local_device_count()))
  log_for_0('global batch_size: {}'.format(config.batch_size))
  train_loader, steps_per_epoch = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='train',
    # split='val',
  )
  # train_loader, steps_per_epoch, yierbayiyiliuqi = input_pipeline.create_split(
  #   dataset_builder,
  #   dataset_config=dataset_config,
  #   training_config=config,
  #   local_batch_size=local_batch_size,
  #   input_type=input_type,
  #   train=False if dataset_config.fake_data else True
  # )
  # val_loader, val_steps, _ = create_split(
  #   dataset_builder,
  #   dataset_config=dataset_config,
  #   config=config,
  #   local_batch_size=local_batch_size,
  #   input_type=input_type,
  #   train=False
  # )
  if dataset_config.fake_data:
    log_for_0('Note: using fake data')
  log_for_0('steps_per_epoch: {}'.format(steps_per_epoch))
  # log_for_0('eval_steps: {}'.format(val_steps))

  ########### Create Model ###########
  model_cls = models_ddpm.SimDDPM
  rngs = nn.Rngs(config.seed, params=config.seed + 114, dropout=config.seed + 514, train=config.seed + 1919)
  dtype = get_dtype(config.half_precision)
  model_init_fn = partial(model_cls, num_classes=NUM_CLASSES, dtype=dtype,**config.diffusion)
  model = model_init_fn(rngs=rngs, **model_config)
  show_dict(f'number of model parameters:{count_params(model)}')

  ########### Create LR FN ###########
  base_lr = config.learning_rate
  learning_rate_fn = create_learning_rate_fn(
    config=config,
    base_learning_rate=base_lr,
    steps_per_epoch=steps_per_epoch,
  )

  ema_scales_fn = partial(edm_ema_scales_schedules, steps_per_epoch=steps_per_epoch, config=config)

  ########### Create Train State ###########
  state = create_train_state(config, model, image_size, learning_rate_fn)
  # # restore checkpoint kaiming
  # if config.restore != '':
  #   log_for_0('Restoring from: {}'.format(config.restore))
  #   state = restore_checkpoint(state, config.restore)
  # elif config.pretrain != '':
  #   raise NotImplementedError("其实会写")
  #   log_for_0('Loading pre-trained from: {}'.format(config.restore))
  #   state = restore_pretrained(state, config.pretrain, config)

  # restore checkpoint zhh
  model_avg = None
  if config.load_from is not None:
    if not os.path.isabs(config.load_from):
      raise ValueError('Checkpoint path must be absolute')
    if not os.path.exists(config.load_from):
      raise ValueError('Checkpoint path {} does not exist'.format(config.load_from))
    state, model_avg = restore_checkpoint(model_init_fn, state, config.load_from, model_config, ema=False) # NOTE: whether to use the ema model
    # sanity check, as in Kaiming's code
    assert state.step > 0 and state.step % steps_per_epoch == 0, ValueError('Got an invalid checkpoint with step {}'.format(state.step))
  step_offset = int(state.step)
  epoch_offset = step_offset // steps_per_epoch  # sanity check for resuming
  assert epoch_offset * steps_per_epoch == step_offset

  state = ju.replicate(state) # NOTE: this doesn't split the RNGs automatically, but it is an intended behavior
  if model_avg is None:
    model_avg = state.params
  else:
    model_avg = ju.replicate(model_avg)

  p_train_step_compute = jax.pmap(
    partial(train_step_compute, 
            learning_rate_fn=learning_rate_fn, ema_scales_fn=ema_scales_fn, 
            config=config),
    axis_name='batch'
  )

  vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())  # for visualization
  log_for_0(f'fixed_sample_idx: {vis_sample_idx}')

  ########### FID ###########
  if config.model.ode_solver in ['jax', 'O']:
    p_sample_step = jax.pmap(
      partial(sample_step, 
              model=model, 
              rng_init=random.PRNGKey(0), 
              device_batch_size=config.fid.device_batch_size, 
              config=config,
              option='FID'
      ),
      axis_name='batch'
    )
    p_visualize_sample_step = jax.pmap(
      partial(sample_step, 
              model=model, 
              rng_init=random.PRNGKey(0), 
              device_batch_size=100, # 一行10个刚好
              config=config,
              option='vis'
      ),
      axis_name='batch'
    )

    def run_p_sample_step(p_sample_step_, state, sample_idx):
      """
      state: train state
      """
      # redefine the interface
      images, nfe = p_sample_step_(state, sample_idx=sample_idx)
      # print("In function run_p_sample_step; images.shape: ", images.shape, flush=True)
      jax.random.normal(random.key(0), ()).block_until_ready()
      nfe = nfe.mean()
      return images[0], nfe  # images have been all gathered
  else:
    raise NotImplementedError('Unknown ode_solver: {}'.format(config.model.ode_solver))
  
  # ------------------------------------------------------------------------------------
  if config.fid.on_use:  # we will evaluate fid    
    inception_net = fid_util.build_jax_inception()
    stats_ref = fid_util.get_reference(config.fid.cache_ref, inception_net)

    if config.fid.eval_only: # debug, this is tang

      samples_all, _ = sample_util.generate_samples_for_fid_eval(state, workdir, config, p_sample_step, run_p_sample_step)
      mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
      fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
      log_for_0(f' w/ ema: FID at {samples_all.shape[0]} samples: {fid_score}')
      return None

    # debugging here
    # samples_dir = '/kmh-nfs-us-mount/logs/kaiminghe/results-edm/edm-cifar10-32x32-uncond-vp'
    # samples = sample_util.get_samples_from_dir(samples_dir, config)
  # ------------------------------------------------------------------------------------

  ########### Training Loop ###########
  # sample_step(state, image_size, sampling_config, epoch_offset, use_wandb=config.wandb, dtype=dtype)
  train_metrics_buffer = []
  train_metrics_last_t = time.time()
  log_for_0('Initial compilation, this might take some minutes...')

  p_update_model_avg = jax.pmap(_update_model_avg, axis_name='batch')
  
  BEST_FID_UNTIL_NOW = config.get('best_fid_until_now', float('inf'))

  for epoch in range(epoch_offset, config.num_epochs):

    ########### Train ###########
    timer = Timer()
    log_for_0('epoch {}...'.format(epoch))
    timer.reset()
    for n_batch, batch in zip(range(steps_per_epoch), train_loader):

      step = epoch * steps_per_epoch + n_batch
      assert config.aug.use_edm_aug == False, "we don't support edm aug for now"
      batch = prepare_batch_data(batch, config)
      # ep = step / steps_per_epoch
      ep = epoch + n_batch / steps_per_epoch # avoid jumping

      # img = batch['image']
      # print(f"img.shape: {img.shape}")
      # print(f'image max: {jnp.max(img)}, min: {jnp.min(img)}') # [-1, 1]
      # img = img * (jnp.array(input_pipeline.STDDEV_RGB)/255.).reshape(1,1,1,3) + (jnp.array(input_pipeline.MEAN_RGB)/255.).reshape(1,1,1,3)
      # print(f"after process, img max: {jnp.max(img)}, min: {jnp.min(img)}")
      # exit(114514)
      # # print("images.shape: ", images.shape)
      # arg_batch, t_batch, target_batch = prepare_batch(batch, rngs, config)

      # print("batch['image'].shape:", batch['image'].shape)
      # assert False

      # # here is code for us to visualize the images
      # import matplotlib.pyplot as plt
      # import numpy as np
      # import os
      # images = batch["image"]
      # print(f"images.shape: {images.shape}", flush=True)

      # from input_pipeline import MEAN_RGB, STDDEV_RGB

      # # save batch["image"] to ./images/{epoch}/i.png
      # rank = jax.process_index()

      # # if os.path.exists(f"/kmh-nfs-us-mount/staging/sqa/images/{n_batch}/{rank}") == False:
      # #   os.makedirs(f"/kmh-nfs-us-mount/staging/sqa/images/{n_batch}/{rank}")
      # path = f"/kmh-nfs-ssd-eu-mount/logs/sqa/flow-matching/sqa_flow-matching/dataset_images/{n_batch}/{rank}"
      # if os.path.exists(path) == False:
      #   os.makedirs(path)
      # for i in range(images[0].shape[0]):
      #   # print the max and min of the image
      #   # print(f"max: {np.max(images[0][i])}, min: {np.min(images[0][i])}")
      #   # img_test = images[0][:100]
      #   # save_img(img_test, f"/kmh-nfs-ssd-eu-mount/code/qiao/flow-matching/sqa_flow-matching/dataset_images/{n_batch}/{rank}", im_name=f"{i}.png", grid=(10, 10))
      #   # break
      #   # use the max and min to normalize the image to [0, 1]
      #   img = images[0][i]
      #   img = img * (jnp.array(STDDEV_RGB)/255.).reshape(1,1,3) + (jnp.array(MEAN_RGB)/255.).reshape(1,1,3)
      #   # print(f"max: {np.max(img)}, min: {np.min(img)}")
      #   img = jnp.clip(img, 0, 1)
      #   # img = (img - np.min(img)) / (np.max(img) - np.min(img))
      #   # img = img.squeeze(-1)
      #   plt.imsave(path+f"/{i}.png", img) # if MNIST, add cmap='gray'
      #   # if i>6: break

      # print(f"saving images for n_batch {n_batch}, done.")
      # if n_batch > 0:
      #   exit(114514)
      # continue

      state, metrics, vis = train_step(state, batch, rngs, p_train_step_compute, config=config)
      
      if epoch == epoch_offset and n_batch == 0:
        log_for_0('p_train_step compiled in {}s'.format(time.time() - train_metrics_last_t))
        log_for_0('Initial compilation completed. Reset timer.')

      if config.get('log_per_step'):
        train_metrics_buffer.append(metrics)
        if (step + 1) % config.log_per_step == 0:
          train_metrics = common_utils.get_metrics(train_metrics_buffer)
          tang_reduce(train_metrics) # do an average
          step_per_sec = config.log_per_step / timer.elapse_with_reset()
          loss_to_display = train_metrics['loss_train']
          if config.wandb and index == 0:
            wandb.log({
              'ema_decay': train_metrics['ema_decay'],
              'ep:': ep, 
              'loss_train': loss_to_display, 
              'lr': train_metrics['lr'], 
              'step': step, 
              'step_per_sec': step_per_sec})
          # log_for_0('epoch: {} step: {} loss: {}, step_per_sec: {}'.format(ep, step, loss_to_display, step_per_sec))
          log_for_0(f'step: {step}, loss: {loss_to_display}, step_per_sec: {step_per_sec}')
          train_metrics_buffer = []

      # EMA
      model_avg = p_update_model_avg(model_avg, state.params, ema_decay=ema_scales_fn(step)[0].repeat(jax.local_device_count()))

      # break
    ########### Save Checkpt ###########
    # we first save checkpoint, then do eval. Reasons: 1. if eval emits an error, then we still have our model; 2. avoid the program exits before the checkpointer finishes its job.
    # NOTE: when saving checkpoint, should sync batch stats first.

    # zhh's checkpointer
    if (
      (epoch + 1) % config.checkpoint_per_epoch == 0
      or epoch == config.num_epochs
      or epoch == 0  # saving at the first epoch for sanity check
      ):
      # pass
      # if index == 0:
      state = sync_batch_stats(state)
      if not config.get('save_by_fid', False):
        save_checkpoint(state, workdir, model_avg)
    if epoch == config.num_epochs - 1:
      state = state.replace(params=model_avg)

    # # Kaiming's checkpointer
    # if (
    #   (epoch + 1) % config.checkpoint_per_epoch == 0
    #   or epoch == config.num_epochs
    #   or epoch == 0  # saving at the first epoch for sanity check
    # ):
    #   state = sync_batch_stats(state)
    #   # TODO{km}: suppress the annoying warning.
    #   save_checkpoint(state, workdir)
    #   log_for_0(f'Work dir: {workdir}')  # for monitoring

    # logging visualizations
    if (epoch + 1) % config.visualize_per_epoch == 0:
      vis = visualize_cifar_batch(vis)
      # print("vis.shape: ", vis.shape) # (8, 160, 256, 3)
      vis = jax.device_get(vis)
      vis = vis[0]
      canvas = Image.fromarray(vis)
      if config.wandb and index == 0:
        wandb.log({'visualize': wandb.Image(canvas)})

    ########### Sampling ###########
    if (epoch + 1) % config.eval_per_epoch == 0:
      log_for_0(f'Sample epoch {epoch}...')
      # sync batch statistics across replicas
      eval_state = sync_batch_stats(state)
      # eval_state = eval_state.replace(params=model_avg)
      vis, _ = run_p_sample_step(p_visualize_sample_step, eval_state, vis_sample_idx)
      vis = make_grid_visualization(vis,grid=10,max_bz=10)
      vis = jax.device_get(vis) # np.ndarray
      vis = vis[0]
      # print(vis.shape)
      # exit("王广廷")
      canvas = Image.fromarray(vis)
      if index == 0:
        if config.wandb:
          wandb.log({'gen': wandb.Image(canvas)})
        else:
          canvas.save(os.path.join(workdir, f'epoch_{epoch}_sample.png'))
      # sample_step(eval_state, image_size, sampling_config, epoch, use_wandb=config.wandb)

    ########### FID ###########
    if config.fid.on_use and (
      (epoch + 1) % config.fid.fid_per_epoch == 0
      or epoch == config.num_epochs
      # or epoch == 0
    ):
      samples_all, _ = sample_util.generate_samples_for_fid_eval(state, workdir, config, p_sample_step, run_p_sample_step)
      mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
      fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
      log_for_0(f'w/o ema: FID at {samples_all.shape[0]} samples: {fid_score}')

      # ema results are much better
      eval_state = sync_batch_stats(state)
      eval_state = eval_state.replace(params=model_avg)
      samples_all, nfe = sample_util.generate_samples_for_fid_eval(eval_state, workdir, config, p_sample_step, run_p_sample_step)
      mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
      fid_score_ema = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
      log_for_0(f'w/ ema: FID at {samples_all.shape[0]} samples: {fid_score_ema}')

      if config.wandb and rank == 0:
        wandb.log({
          'FID': fid_score,
          'FID_ema': fid_score_ema,
          'NFE': nfe
        })

      vis = make_grid_visualization(samples_all, to_uint8=False)
      vis = jax.device_get(vis)
      vis = vis[0]
      canvas = Image.fromarray(vis)
      if config.wandb and index == 0:
        wandb.log({'gen_fid': wandb.Image(canvas)})
    
      if config.get('save_by_fid', False):
        if fid_score_ema < BEST_FID_UNTIL_NOW:
          BEST_FID_UNTIL_NOW = fid_score_ema
          # import shutil
          # if os.path.exists(os.path.join(workdir, 'best_fid')):
          #   shutil.rmtree(os.path.join(workdir, 'best_fid'))
          os.makedirs(os.path.join(workdir, 'best_fid'), exist_ok=True)
          if index == 0:
            with open(os.path.join(workdir,'best_fid', 'FID.txt'), 'w') as f:
              f.write(str(BEST_FID_UNTIL_NOW))
          save_checkpoint(state, os.path.join(workdir, 'best_fid'), model_avg)
          log_for_0(f'[BEST FID HAS CHANGED]: {BEST_FID_UNTIL_NOW}')

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  if index == 0 and config.wandb:
    wandb.finish()

  return state

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
    wandb.init(project='LMCI-eval', dir=workdir, tags=['Sanity_Check'])
    # wandb.init(project='sqa_edm_debug', dir=workdir)
    wandb.config.update(config.to_dict())
  # dtype = jnp.bfloat16 if model_config.half_precision else jnp.float32
  global_seed(config.seed)
  image_size = model_config.image_size

  ########### Create Model ###########
  model_cls = models_ddpm.SimDDPM
  rngs = nn.Rngs(config.seed, params=config.seed + 114, dropout=config.seed + 514, train=config.seed + 1919)
  dtype = get_dtype(config.half_precision)
  # model_init_fn = partial(model_cls, num_classes=NUM_CLASSES, dtype=dtype)
  model_init_fn = partial(model_cls, num_classes=NUM_CLASSES, dtype=dtype, **config.diffusion)
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
  state,_ = restore_checkpoint(model_init_fn, state, config.load_from, model_config, ema=config.evalu.ema) # NOTE: whether to use the ema model
  state_step = int(state.step)
  state = ju.replicate(state) # NOTE: this doesn't split the RNGs automatically, but it is an intended behavior

  
  # ### debug sampler here
  # assert False, 'Please note that you should delete the "replicate" line'
  # num_steps = 1000
  # # state = state[0]
  # t = model.compute_t(jnp.arange(num_steps), num_steps)
  # vis, denoised = generate(state, model, random.PRNGKey(0), 1) # (num_steps, 32, 32, 3)
  # print("vis.shape: ", vis.shape)
  # vis = vis.reshape(num_steps, 32, 32, 3)
  # denoised = denoised.reshape(num_steps, 32, 32, 3)
  # assert vis.shape == (num_steps, 32, 32, 3)
  # assert denoised.shape == (num_steps, 32, 32, 3)
  # from utils.vis_util import float_to_uint8
  # for ep in range(num_steps):
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
  if config.model.ode_solver in ['jax', 'O']:
    p_sample_step = jax.pmap(
      partial(sample_step, 
              model=model, 
              rng_init=random.PRNGKey(0), 
              device_batch_size=config.fid.device_batch_size,               config=config,
              option='FID'
      ),
      axis_name='batch'
    )
    p_visualize_sample_step = jax.pmap(
      partial(sample_step, 
              model=model, 
              rng_init=random.PRNGKey(0), 
              device_batch_size=100, 
              config=config,
              option='vis',
              # MEAN_RGB=input_pipeline.MEAN_RGB, 
              # STDDEV_RGB=input_pipeline.STDDEV_RGB
      ),
      axis_name='batch'
    )

    def run_p_sample_step(p_sample_step_, state, sample_idx):
      """
      state: train state
      """
      # redefine the interface
      images, nfe = p_sample_step_(state, sample_idx=sample_idx)
      # print("In function run_p_sample_step; images.shape: ", images.shape, flush=True)
      jax.random.normal(random.key(0), ()).block_until_ready()
      # print('images.shape:',jax.device_get(images).shape)
      nfe = nfe.mean()
      return images[0], nfe  # images have been all gathered
  
  else:
    raise NotImplementedError('Unknown ode_solver: {}'.format(config.model.ode_solver))
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
    vis, nfe = run_p_sample_step(p_visualize_sample_step, eval_state, vis_sample_idx)
    vis = make_grid_visualization(vis,grid=10,max_bz=10)
    vis = jax.device_get(vis) # np.ndarray
    vis = vis[0]
    # print(vis.shape)
    # exit("王广廷")
    canvas = Image.fromarray(vis)
    if config.wandb and index == 0:
      wandb.log({'gen': wandb.Image(canvas)})
    log_for_0('Sample NFE: {}'.format(nfe))
    # assert False, 'image saved!: {}'.format(nfe)
    # sample_step(eval_state, image_size, sampling_config, epoch, use_wandb=config.wandb)
  ########### FID ###########
  if config.fid.on_use:

    samples_all, nfe = sample_util.generate_samples_for_fid_eval(eval_state, workdir, config, p_sample_step, run_p_sample_step)
    # assert samples_all.ndim == 10086, 'Get wrong shape: {}'.format(samples_all.shape)
    # print("samples_all.shape: ", samples_all.shape)
    mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
    fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
    log_for_0(f'FID at {samples_all.shape[0]} samples: {fid_score}')

    if config.wandb and rank == 0:
      wandb.log({
        'FID': fid_score,
        'NFE': nfe
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