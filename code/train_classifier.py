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

# Pater noster, qui es in caelis,
# sanctificetur nomen tuum.
# Adveniat regnum tuum.
# Fiat voluntas tua, sicut in caelo, et in terra.
# Panem nostrum quotidianum da nobis hodie,
# et dimitte nobis debita nostra,
# sicut et nos dimittimus debitoribus nostris.
# Et ne nos inducas in tentationem,
# sed libera nos a malo.
# Amen.
# import models.models_ddpm as models_ddpm
import models.models_ddpm_classifier as models_ddpm_classifier
from models.models_ddpm import edm_ema_scales_schedules, diffusion_schedule_fn_some, create_zhh_SAMPLING_diffusion_schedule
# from models.models_ddpm_classifier import generate, edm_ema_scales_schedules, diffusion_schedule_fn_some, create_zhh_SAMPLING_diffusion_schedule
from input_pipeline_classifier import prepare_batch_data, create_split

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
  # raise NotImplementedError('Another LR scchedule!')
  """Create learning rate schedule."""
  if config.lr_schedule == 'classifier_specific':
    return optax.linear_schedule(base_learning_rate, 0, config.num_epochs * steps_per_epoch)
  
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


def train_step_compute(state: NNXTrainState, batch, noise_batch, t_batch, learning_rate_fn, ema_scales_fn,diffusion_schedule_fn, config,really_is_train=True):
  """
  Perform a single training step.
  We will pmap this function
  ---
  batch: a dict, with image, label, augment_label
  noise_batch: the noise_batch for the model
  t_batch: the t_batch for the model
  """

  ema_decay, scales = ema_scales_fn(state.step)
  
  # use "diffusion_schedule_fn" to process t_batch
  alpha_cumprod_batch, beta_batch, alpha_cumprod_prev_batch, posterior_log_variance_clipped_batch = diffusion_schedule_fn(t_batch=t_batch)

  def loss_fn(params_to_train):
    """loss function used for training."""
    
    outputs = state.apply_fn(state.graphdef, params_to_train, state.rng_states, state.batch_stats, state.useless_variable_state, True, batch['image'], batch['label'], batch['augment_label'], noise_batch, t_batch, alpha_cumprod_batch=alpha_cumprod_batch, beta_batch=beta_batch,alpha_cumprod_prev_batch=alpha_cumprod_prev_batch,posterior_log_variance_clipped_batch=posterior_log_variance_clipped_batch)
    loss, new_batch_stats, new_rng_states, dict_losses, images = outputs

    return loss, (new_batch_stats, new_rng_states, dict_losses, images)

  step = state.step
  lr = learning_rate_fn(step)

  if really_is_train:
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      # aux, grads = grad_fn(state.params)
      aux, grads = grad_fn(state.params)
      # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
      grads = lax.pmean(grads, axis_name='batch')

      new_batch_stats, new_rng_states, dict_losses, images = aux[1]
      
      new_state = state.apply_gradients(
        grads=grads, batch_stats=new_batch_stats, rng_states=new_rng_states
      )
  else: # actually, is val
      loss, aux = loss_fn(state.params)
      new_batch_stats, new_rng_states, dict_losses, images = aux
      new_state = state
      
  
  metrics = compute_metrics(dict_losses)
  metrics['lr'] = lr
  metrics['ema_decay'] = ema_decay
  metrics['scales'] = scales

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
  # t_batch = jax.random.uniform(rngs.train(), (b1, b2))
  t_batch = jax.random.randint(rngs.train(), (b1, b2), minval=0, maxval=config.diffusion_nT) # [0, num_time_steps)

  new_state, metrics, images = train_step_compute_fn(state, batch, noise_batch, t_batch)

  return new_state, metrics, images

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
  assert ema == False, 'we donnot use ema for now'
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
    # 'ema_mo_xing': useful_abs_state,
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
def save_checkpoint(state:NNXTrainState, workdir, ):
  state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
  # model_avg = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model_avg))
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
    # 'ema_mo_xing': model_avg,
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

  def apply_fn(graphdef2, params2, rng_states2, batch_stats2, useless_, is_training, images, labels, augment_labels, noise_batch, t_batch,alpha_cumprod_batch,beta_batch,alpha_cumprod_prev_batch,posterior_log_variance_clipped_batch):
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
    loss_train, dict_losses, images = merged_model.forward(images, labels, augment_labels, noise_batch, t_batch, alpha_cumprod_batch=alpha_cumprod_batch, beta_batch=beta_batch, alpha_cumprod_prev_batch=alpha_cumprod_prev_batch, posterior_log_variance_clipped_batch=posterior_log_variance_clipped_batch)
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

## Switch to Pytorch Loader


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
  print('train classifier!')

  ########### Initialize ###########
  rank = index = jax.process_index()
  config.dataset.out_channels = config.model.out_channels
  model_config = config.model 
  dataset_config = config.dataset
  fid_config = config.fid
  if rank == 0 and config.wandb:
    wandb.init(project='LMCI', dir=workdir, tags=['ADM', 'classifier'])
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

  assert config.batch_size % jax.process_count() == 0, ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()
  assert local_batch_size % jax.local_device_count() == 0, ValueError('Local batch size must be divisible by the number of local devices')
  log_for_0('local_batch_size: {}'.format(local_batch_size))
  log_for_0('jax.local_device_count: {}'.format(jax.local_device_count()))
  log_for_0('global batch_size: {}'.format(config.batch_size))
  train_loader, steps_per_epoch = create_split(
    config.dataset,
    local_batch_size,
    split='train',
  )
  val_loader, val_steps_per_epoch = create_split(
    config.dataset,
    local_batch_size,
    split='val',
  )
  
  if dataset_config.fake_data:
    log_for_0('Note: using fake data')
  log_for_0('steps_per_epoch: {}'.format(steps_per_epoch))
  # log_for_0('eval_steps: {}'.format(val_steps))

  ########### Create Model ###########
  model_cls = models_ddpm_classifier.SimDDPM
  rngs = nn.Rngs(config.seed, params=config.seed + 114, dropout=config.seed + 514, train=config.seed + 1919)
  dtype = get_dtype(config.half_precision)
  # model_init_fn = partial(model_cls, num_classes=NUM_CLASSES, dtype=dtype)
  model_init_fn = partial(model_cls, num_classes=NUM_CLASSES, dtype=dtype, **config.diffusion_schedule)
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
  diffusion_schedule_fn = partial(diffusion_schedule_fn_some, config=config)

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
  if config.load_from is not None:
    if not os.path.isabs(config.load_from):
      raise ValueError('Checkpoint path must be absolute')
    if not os.path.exists(config.load_from):
      raise ValueError('Checkpoint path {} does not exist'.format(config.load_from))
    state = restore_checkpoint(model_init_fn ,state, config.load_from)
    # sanity check, as in Kaiming's code
    assert state.step > 0 and state.step % steps_per_epoch == 0, ValueError('Got an invalid checkpoint with step {}'.format(state.step))
  step_offset = int(state.step)
  epoch_offset = step_offset // steps_per_epoch  # sanity check for resuming
  assert epoch_offset * steps_per_epoch == step_offset

  state = ju.replicate(state) # NOTE: this doesn't split the RNGs automatically, but it is an intended behavior
  # model_avg = state.params

  p_train_step_compute = jax.pmap(
    partial(train_step_compute, 
            learning_rate_fn=learning_rate_fn, ema_scales_fn=ema_scales_fn, 
            diffusion_schedule_fn = diffusion_schedule_fn,
            really_is_train=True,
            config=config),
    axis_name='batch'
  )
  p_val_step_compute = jax.pmap(
    partial(train_step_compute, 
            learning_rate_fn=learning_rate_fn, ema_scales_fn=ema_scales_fn, 
            diffusion_schedule_fn = diffusion_schedule_fn,
            really_is_train=False,
            config=config),
    axis_name='batch'
  )

  vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())  # for visualization
  log_for_0(f'fixed_sample_idx: {vis_sample_idx}')

  
  ########### Training Loop ###########
  # sample_step(state, image_size, sampling_config, epoch_offset, use_wandb=config.wandb, dtype=dtype)
  train_metrics_buffer = []
  train_metrics_last_t = time.time()
  log_for_0('Initial compilation, this might take some minutes...')

  # p_update_model_avg = jax.pmap(_update_model_avg, axis_name='batch')
  
  # NOTE: to avoid tracedarray error, we first do some random things here
  ### GOD BLESS US ###
  # Pater noster, qui es in caelis,
  # sanctificetur nomen tuum.
  # Adveniat regnum tuum.
  # Fiat voluntas tua, sicut in caelo, et in terra.
  # Panem nostrum quotidianum da nobis hodie,
  # et dimitte nobis debita nostra,
  # sicut et nos dimittimus debitoribus nostris.
  # Et ne nos inducas in tentationem,
  # sed libera nos a malo.
  # Amen.
  ### GOD BLESS US ###
  create_zhh_SAMPLING_diffusion_schedule(config=config)

  for epoch in range(epoch_offset, config.num_epochs):

    ########### Train ###########
    timer = Timer()
    log_for_0('epoch {}...'.format(epoch))
    timer.reset()
    for n_batch, batch in zip(range(steps_per_epoch), train_loader):

      step = epoch * steps_per_epoch + n_batch
      assert config.aug.use_edm_aug == False, "we don't support edm aug for now"
      batch = prepare_batch_data(batch, config)
      # batch['label'].shape: (b1, b2), each element is 0-9
      # ep = step * config.batch_size / yierbayiyiliuqi
      ep = step / steps_per_epoch

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
          acc_to_display = train_metrics['acc_train']
          if config.wandb and index == 0:
            wandb.log({
              # 'ema_decay': train_metrics['ema_decay'],
              'ep:': ep, 
              'loss_train': loss_to_display, 
              'acc_train': acc_to_display,
              'lr': train_metrics['lr'], 
              'step': step, 
              'step_per_sec': step_per_sec})
          # log_for_0('epoch: {} step: {} loss: {}, step_per_sec: {}'.format(ep, step, loss_to_display, step_per_sec))
          log_for_0(f'step: {step}, loss: {loss_to_display}, acc: {acc_to_display}, step_per_sec: {step_per_sec}')
          train_metrics_buffer = []

      # EMA
      # raise LookupError("真有EMA吗？")
      # model_avg = p_update_model_avg(model_avg, state.params, ema_decay=ema_scales_fn(step)[0].repeat(jax.local_device_count()))

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
      save_checkpoint(state, workdir, )
    # if epoch == config.num_epochs - 1:
    #   state = state.replace(params=model_avg)

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

    ########### Eval ###########
    state = sync_batch_stats(state)
    val_metrics_buffer = []
    for n_batch, batch in zip(range(val_steps_per_epoch), val_loader):

      step = epoch * steps_per_epoch + n_batch
      assert config.aug.use_edm_aug == False, "we don't support edm aug for now"
      batch = prepare_batch_data(batch, config)
      # batch['label'].shape: (b1, b2), each element is 0-9
      # ep = step * config.batch_size / yierbayiyiliuqi
      ep = step / steps_per_epoch

      state, metrics, vis = train_step(state, batch, rngs, p_val_step_compute, config=config)
      
      val_metrics_buffer.append(metrics)
    
      if (step + 1) % config.log_per_step == 0:
        log_for_0('Eval step: {} / {}'.format(step, val_steps_per_epoch))
    
    val_metrics = common_utils.get_metrics(val_metrics_buffer)
    tang_reduce(val_metrics) # do an average
    step_per_sec = config.log_per_step / timer.elapse_with_reset()
    loss_to_display = val_metrics['loss_train']
    acc_to_display = val_metrics['acc_train']
    if config.wandb and index == 0:
      wandb.log({
        'ep:': ep, 
        'loss_val': loss_to_display, 
        'acc_val': acc_to_display,
        'step': step
      })
    # log_for_0('epoch: {} step: {} loss: {}, step_per_sec: {}'.format(ep, step, loss_to_display, step_per_sec))
    log_for_0(f'[Eval] step: {step}, loss: {loss_to_display}, acc: {acc_to_display}, step_per_sec: {step_per_sec}')
    
    # logging visualizations
    if (epoch + 1) % config.visualize_per_epoch == 0:
      vis = visualize_cifar_batch(vis)
      # print("vis.shape: ", vis.shape) # (8, 160, 256, 3)
      vis = jax.device_get(vis)
      vis = vis[0]
      canvas = Image.fromarray(vis)
      if index == 0:
        # raise LookupError("想个办法visualize 分类，也许不重要")
        if config.wandb:
          wandb.log({'visualize': wandb.Image(canvas)})
        else:
          canvas.save(os.path.join(workdir, f'visualize_{epoch}.png'))

  
  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  if index == 0 and config.wandb:
    wandb.finish()

  return state

def just_evaluate(
    config: ml_collections.ConfigDict, workdir: str
  ):
  raise NotImplementedError