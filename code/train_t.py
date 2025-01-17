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

# import models.models_ddpm as models_ddpm
import models.t.t as t
from models.models_ddpm import generate, edm_ema_scales_schedules
from input_pipeline import create_split

NUM_CLASSES = 10

def get_input_pipeline(dataset_config):
    assert dataset_config.name == 'cifar10'
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

这个函数其实用到了 = LookupError('LookupError:看上去有错误')

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


def train_step_compute(state: NNXTrainState, batch, noise_batch, t_batch, learning_rate_fn, config):
  """
  Perform a single training step.
  We will pmap this function
  ---
  batch: a dict, with image, label, augment_label
  noise_batch: the noise_batch for the model
  t_batch: the t_batch for the model
  """

  def loss_fn(params_to_train):
    """loss function used for training."""
    
    outputs = state.apply_fn(state.graphdef, params_to_train, state.rng_states, state.batch_stats, state.useless_variable_state, True, batch['image'], noise_batch, t_batch)
    sigma_pred, new_batch_stats, new_rng_states = outputs
    # jax.debug.print('t_pred.shape: {s}', s=t_pred.shape)
    # jax.debug.print('t_batch.shape: {s}', s=t_batch.shape)
    ### EDM
    t_pred = 1 / (1 + sigma_pred)
    t_batch = 1 / (1 + t_batch)
    loss = jnp.mean((t_pred - t_batch.reshape(-1,1)) ** 2) # 菜就多练
    dict_losses = {'loss': loss}

    return loss, (new_batch_stats, new_rng_states, dict_losses, sigma_pred) # for debug

  step = state.step
  dynamic_scale = None
  lr = learning_rate_fn(step)

  if dynamic_scale:
    raise NotImplementedError
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name='batch')

  # for simplicity, we don't all gather images
  # loss = aux[0]
  new_batch_stats, new_rng_states, dict_losses, sigma_pred = aux[1]
  metrics = compute_metrics(dict_losses)
  metrics['lr'] = lr

  new_state = state.apply_gradients(
    grads=grads, batch_stats=new_batch_stats, rng_states=new_rng_states
  )

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

  return new_state, metrics, sigma_pred


def train_step(state: NNXTrainState, batch, rngs, train_step_compute_fn):
  """
  Perform a single training step.
  We will NOT pmap this function
  ---
  batch: a dict, with image, label, augment_label
  rngs: nnx.Rngs
  train_step_compute_fn: the pmaped version of train_step_compute
  """

  images = batch['image']
  # print("images.shape: ", images.shape) # (8, 64, 32, 32, 3)
  b1, b2 = images.shape[0], images.shape[1]
  noise_batch = jax.random.normal(rngs.train(), images.shape)
  # t_batch = jax.random.uniform(rngs.train(), (b1, b2))
  # EDM: t_batch is normal
  t_batch = jax.random.normal(rngs.train(), (b1, b2)) 
  t_batch = jnp.exp(t_batch * 1.2 - 1.2)

  new_state, metrics, t_pred = train_step_compute_fn(state, batch, noise_batch, t_batch)
  # print("t_pred.shape: ", t_pred.shape)
  # print(t_pred[0][:4].reshape(4))

  return new_state, metrics

def sqa_eval_step(state: NNXTrainState, batch, rngs, train_step_compute_fn, eval_noise_scale):
  """
  Perform a single training step.
  We will NOT pmap this function
  ---
  batch: a dict, with image, label, augment_label
  rngs: nnx.Rngs
  train_step_compute_fn: the pmaped version of train_step_compute
  """

  images = batch['image']
  # print("images.shape: ", images.shape) # (8, 64, 32, 32, 3)
  b1, b2 = images.shape[0], images.shape[1]
  noise_batch = jax.random.normal(rngs.train(), images.shape)
  # EDM: t_batch is normal
  t_batch = jax.random.normal(rngs.train(), (b1, b2)) 
  t_batch = jnp.exp(t_batch * 1.2 - 1.2)

  new_state, metrics, t_pred = train_step_compute_fn(state, batch, noise_batch, t_batch)
  # print("t_pred.shape: ", t_pred.shape)
  # print(t_pred[0][:4].reshape(4))

  return new_state, metrics

def eval_step(train_state:NNXTrainState, batch, noise_batch, t_batch):
  sigma_pred, new_batch_stats, new_rng_params = train_state.apply_fn(train_state.graphdef, train_state.params, train_state.rng_states, train_state.batch_stats, train_state.useless_variable_state, False, batch['image'], noise_batch, t_batch) # False: is_training
  ### EDM
  t_pred = 1 / (1 + sigma_pred)
  t_batch = 1 / (1 + t_batch)
  loss = jnp.mean((t_pred - t_batch.reshape(-1,1)) ** 2) # 菜就多练
  dict_losses = {'loss': loss}

  # compute metrics
  metrics = compute_metrics(dict_losses)

  return None, metrics, sigma_pred # for train_step use

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

def restore_checkpoint(model_init_fn, state, workdir, model_config):
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
    'you_hua_qi': state.opt_state,
    'step': 0
  }
  loaded_state = checkpoints.restore_checkpoint(workdir, target=fake_state,orbax_checkpointer=checkpointer)
  merged_params = loaded_state['mo_xing']
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
def save_checkpoint(state:NNXTrainState, workdir, model_avg=None):
  """
  model_avg: not used
  """
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

  def apply_fn(graphdef2, params2, rng_states2, batch_stats2, useless_, is_training, images, noise_batch, t_batch):
    """
    input:
      images
      noise
      t
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
    t = t_batch.reshape(-1, 1, 1, 1)
    noisy = images + noise_batch * t
    c_in = 1 / jnp.sqrt(t ** 2 + 0.5 ** 2)
    t_pred = merged_model.forward(c_in * noisy)
    # for debug
    # t_pred = merged_model.forward(jnp.ones_like(noise_batch) * t)
    new_batch_stats, new_rng_states, _ = nn.state(merged_model, nn.BatchStat, nn.RngState, ...)
    return t_pred, new_batch_stats, new_rng_states

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

  Useless here
  """

  return batch

# def _update_model_avg(model_avg, state_params, ema_decay):
#   return jax.tree_util.tree_map(lambda x, y: ema_decay * x + (1.0 - ema_decay) * y, model_avg, state_params)
#   # return model_avg

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
    wandb.init(project='sqa_t_pred', dir=workdir, tags=['sqa_EDM'])
    wandb.config.update(config.to_dict())
  global_seed(config.seed)

  image_size = model_config.image_size

  log_for_0('config.batch_size: {}'.format(config.batch_size))

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

  ########### Create Model ###########
  model_cls = getattr(t, model_config.net_type)
  rngs = nn.Rngs(config.seed, params=config.seed + 114, dropout=config.seed + 514, train=config.seed + 1919)
  dtype = get_dtype(config.half_precision)
  model_init_fn = partial(model_cls, dtype=dtype)
  model = model_init_fn(rngs=rngs, **model_config)
  num_params = count_params(model)["useful_params"]
  log_for_0(f'number of model parameters:{num_params}')
  if config.wandb and index == 0:
    wandb.log({'num_params': num_params})

  ########### Create LR FN ###########
  # base_lr = config.learning_rate * config.batch_size / 256.
  base_lr = config.learning_rate
  learning_rate_fn = create_learning_rate_fn(
    config=config,
    base_learning_rate=base_lr,
    steps_per_epoch=steps_per_epoch,
  )

  ########### Create Train State ###########
  state = create_train_state(config, model, image_size, learning_rate_fn)

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
  model_avg = state.params


  p_train_step_compute = jax.pmap(
    partial(train_step_compute, 
            learning_rate_fn=learning_rate_fn, 
            config=config),
    axis_name='batch'
  )

  p_eval_step_compute = jax.pmap(
    eval_step,
    axis_name='batch'
  )

  ########### Training Loop ###########
  train_metrics_buffer = []
  train_metrics_last_t = time.time()
  log_for_0('Initial compilation, this might take some minutes...')

  for epoch in range(epoch_offset, config.num_epochs):

    ########### Train ###########
    timer = Timer()
    log_for_0('epoch {}...'.format(epoch))
    timer.reset()
    for n_batch, batch in zip(range(steps_per_epoch), train_loader):

      step = epoch * steps_per_epoch + n_batch
      assert config.aug.use_edm_aug == False, "we don't support edm aug for now"
      batch = prepare_batch_data(batch, config)
      ep = step * config.batch_size / yierbayiyiliuqi

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

      state, metrics = train_step(state, batch, rngs, p_train_step_compute)
      
      if epoch == epoch_offset and n_batch == 0:
        log_for_0('p_train_step compiled in {}s'.format(time.time() - train_metrics_last_t))
        log_for_0('Initial compilation completed. Reset timer.')

      if config.get('log_per_step'):
        train_metrics_buffer.append(metrics)
        if (step + 1) % config.log_per_step == 0:
          train_metrics = common_utils.get_metrics(train_metrics_buffer)
          tang_reduce(train_metrics) # do an average
          step_per_sec = config.log_per_step / timer.elapse_with_reset()
          loss_to_display = train_metrics['loss']
          if config.wandb and index == 0:
            wandb.log({
              'ep:': ep, 
              'loss_train': loss_to_display, 
              'lr': train_metrics['lr'], 
              'step': step, 
              'step_per_sec': step_per_sec})
          # log_for_0('epoch: {} step: {} loss: {}, step_per_sec: {}'.format(ep, step, loss_to_display, step_per_sec))
          log_for_0(f'step: {step}, loss: {loss_to_display}, step_per_sec: {step_per_sec}')
          train_metrics_buffer = []

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
      save_checkpoint(state, workdir, model_avg)

    ########### Evaluation ###########
    log_for_0('Epoch {} evaluation'.format(epoch))
    eval_metrics_buffer = []
    for n_batch, batch in zip(range(val_steps), val_loader):
      batch = prepare_batch_data(batch, config)
      _, metrics = train_step(state, batch, rngs, p_eval_step_compute)
      eval_metrics_buffer.append(metrics)
      if (n_batch + 1) % config.log_per_step == 0:
        log_for_0('epoch: {} [Eval] {}/{}'.format(epoch, n_batch + 1, val_steps))
    eval_metrics = common_utils.get_metrics(eval_metrics_buffer)
    tang_reduce(eval_metrics)
    loss_to_display = eval_metrics['loss']
    if config.wandb and index == 0:
      wandb.log({'test_loss': loss_to_display})
    log_for_0('epoch: [Eval] {} (step {}) test_loss: {} '.format(ep, step, loss_to_display))

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  if index == 0 and config.wandb:
    wandb.finish()

  return state

def just_evaluate(
    config: ml_collections.ConfigDict, workdir: str
  ):
  raise DeprecationWarning
  # from langevin import langevin_step
  ########### Initialize ###########
  rank = index = jax.process_index()
  config.dataset.out_channels = config.model.out_channels
  model_config = config.model 
  dataset_config = config.dataset
  fid_config = config.fid
  if rank == 0 and config.wandb:
    wandb.init(project='sqa_t_eval', dir=workdir)
    wandb.config.update(config.to_dict())
  # dtype = jnp.bfloat16 if model_config.half_precision else jnp.float32
  global_seed(config.seed)
  image_size = model_config.image_size

  ########### Create Model ###########
  model_cls = t.sqa_t_ver1
  rngs = nn.Rngs(config.seed, params=config.seed + 114, dropout=config.seed + 514, train=config.seed + 1919, sample=config.seed + 810)
  # dtype = get_dtype(config.half_precision)
  # model_init_fn = partial(model_cls, dtype=dtype)
  model = model_cls(rngs=rngs)
  show_dict(f'number of model parameters:{count_params(model)}')

  ########### Create LR FN ###########
  learning_rate_fn = lambda:1 # just in order to create the state

  ########### Create Train State ###########
  state = create_train_state(config, model, image_size, learning_rate_fn)
  # assert config.get('load_from',None) is not None, 'Must provide a checkpoint path for evaluation'
  # if not os.path.isabs(config.load_from):
  #   raise ValueError('Checkpoint path must be absolute')
  # if not os.path.exists(config.load_from):
  #   raise ValueError('Checkpoint path {} does not exist'.format(config.load_from))
  state = restore_checkpoint(model_cls, state, "/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20241128_031750_8xab8k_kmh-tpuvm-v2-32-preemptible-2__b_lr_ep_eval/checkpoint_4850", {})
  state_step = int(state.step)
  state = ju.replicate(state) # NOTE: this doesn't split the RNGs automatically, but it is an intended behavior
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
  # train_loader, steps_per_epoch, yierbayiyiliuqi = input_pipeline.create_split(
  #   dataset_builder,
  #   dataset_config=dataset_config,
  #   training_config=config,
  #   local_batch_size=local_batch_size,
  #   input_type=input_type,
  #   train=False if dataset_config.fake_data else True
  # )
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
  # log_for_0('steps_per_epoch: {}'.format(steps_per_epoch))
  log_for_0('eval_steps: {}'.format(val_steps))

  p_eval_step_compute = jax.pmap(
    eval_step,
    axis_name='batch'
  )

  ########### FID ###########
  # no

  ########### Gen ###########

  log_for_0('Eval...')

  ########### Evaluation ###########
  eval_metrics_buffer = []
  eval_scales=[0, 0.01, 0.02, 0.1, 0.2, 0.3]
  for n_batch, batch in zip(range(val_steps), val_loader):
    batch = prepare_batch_data(batch, config) 
    if eval_scales == []: break
    eval_noise_scale = eval_scales[0]
    _, metrics = sqa_eval_step(state, batch, rngs, p_eval_step_compute, eval_noise_scale=eval_noise_scale)
    eval_metrics_buffer.append(metrics)
    # if (n_batch + 1) % config.log_per_step == 0:
    # log_for_0(f"[Eval] noise level {eval_noise_scale}")
    eval_metrics = common_utils.get_metrics(eval_metrics_buffer)
    tang_reduce(eval_metrics)
    loss_to_display = eval_metrics['loss']
    if config.wandb and index == 0:
      wandb.log({
        'test_loss': loss_to_display,
        'eval_noise_scale': eval_noise_scale,
        })
    log_for_0(f'eval noise scale: {eval_noise_scale}, test_loss: {loss_to_display}')
    eval_metrics_buffer = []
    eval_scales.pop(0)

  ########### Sampling ###########
  # # eval_state = sync_batch_stats(state)
  # x = jax.random.normal(jax.random.PRNGKey(0), (8, image_size, image_size, model_config.out_channels))
  # # all_images = [x] # how to visualize?
  # for i in range(config.evalu.epochs):
  #   x, mean_of_t, grad_norm, signal_to_noise_ratio = langevin_step(x, state, rngs, config.evalu)
  #   # all_images.append(x)
  #   y = make_grid_visualization(x)
  #   y = jax.device_get(y)
  #   y = y[0]
  #   canvas = Image.fromarray(y)
  #   if config.wandb and index == 0:
  #     wandb.log({'gen': wandb.Image(canvas)})
  #   log_for_0(f'epoch {i} mean_of_t: {mean_of_t}, grad_norm: {grad_norm}, signal_to_noise_ratio: {signal_to_noise_ratio}')
  #   if config.wandb and index == 0:
  #     wandb.log({
  #       'mean_of_t': mean_of_t,
  #       'grad_norm': grad_norm,
  #       'signal_to_noise_ratio': signal_to_noise_ratio,
  #     })
  ########### FID ###########
  # if config.fid.on_use:

  #   samples_all = sample_util.generate_samples_for_fid_eval(eval_state, workdir, config, p_sample_step, run_p_sample_step)
  #   mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
  #   fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
  #   log_for_0(f'FID at {samples_all.shape[0]} samples: {fid_score}')

  #   if config.wandb and rank == 0:
  #     wandb.log({
  #       'FID': fid_score,
  #     })

  #   vis = make_grid_visualization(samples_all, to_uint8=False)
  #   vis = jax.device_get(vis)
  #   vis = vis[0]
  #   canvas = Image.fromarray(vis)
  #   if config.wandb and index == 0:
  #     wandb.log({'gen_fid': wandb.Image(canvas)})

  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  if index == 0 and config.wandb:
    wandb.finish()