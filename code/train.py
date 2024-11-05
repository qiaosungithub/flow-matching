# Copied from Kaiming He's resnet_jax repository

import time
from typing import Any

from flax.training import checkpoints
from flax.training import dynamic_scale as dynamic_scale_lib
import jax, os, wandb
from jax import lax, random
import jax.numpy as jnp
import ml_collections
import optax

import torch
import numpy as np
from utils.logging_util import log_for_0, Timer
from flax import jax_utils as ju
from utils.metric_utils import tang_reduce, MyMetrics, Avger
from torch.utils.data import DataLoader
from utils.utils import train_set_, val_set_, get_sigmas, save_img, corruption
import ncsnv2

from utils.display_utils import show_dict, display_model, count_params
from functools import partial
from flax.training.train_state import TrainState as FlaxTrainState
import flax.nnx as nn
from langevin import langevin, langevin_masked
from 数据集 import create_split, prepare_batch_data_sqa
from train_step_sqa import train_step_sqa, solve_diffeq

import orbax.checkpoint as ocp
from flax.training import checkpoints

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
    train_config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int,
):
    """Create learning rate schedule."""
    if train_config.scheduler == 'poly':
        return poly_decay_lr_fn(base_learning_rate, train_config.warmup_steps, train_config.num_epochs * steps_per_epoch)
    elif train_config.scheduler == 'constant':
        return constant_lr_fn(base_learning_rate)
    else:
        raise ValueError('Unknown learning rate scheduler {}'.format(train_config.scheduler))

class NNXTrainState(FlaxTrainState):
  batch_stats: Any
  rng_states: Any
  graphdef: Any
  useless_variable_state: Any
  # NOTE: is_training can't be a attr, since it can't be replicated

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

def create_model(*, model_cls, half_precision, config, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(
    dtype=model_dtype, 
    ngf=config.model.ngf, 
    n_noise_levels=config.sampling.n_noise_levels,
    config=config,
    **kwargs)

def criterion(pred, target):
  # L2 lossp
  # return jnp.mean((pred - target) ** 2, axis=0).mean()
  return jnp.mean((pred - target) ** 2, axis=0).mean()

# def restore_checkpoint(state, workdir):
#   return checkpoints.restore_checkpoint(workdir, state)

def restore_checkpoint(model_init_fn, state, workdir):
  abstract_model = nn.eval_shape(lambda: model_init_fn(rngs=nn.Rngs(0)))
  rng_states = state.rng_states
  abs_state = nn.state(abstract_model)
  params, batch_stats, others = abs_state.split(nn.Param, nn.BatchStat, ...)
  useful_abs_state = nn.State.merge(params, batch_stats)
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

checkpointer = ocp.StandardCheckpointer()
def _restore(ckpt_path, item, **restore_kwargs):
  return ocp.StandardCheckpointer.restore(checkpointer, ckpt_path, target=item)
setattr(checkpointer, 'restore', _restore)

# def save_checkpoint(state, workdir):
#   state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
#   step = int(state.step)
#   if jax.process_index() == 0:
#     log_for_0('Saving checkpoint step %d.', step)
#   checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=2)

def save_checkpoint(state:NNXTrainState, workdir):
  state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
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

def get_no_weight_decay_dict(params):
  def modify_value_based_on_key(obj):
    if not isinstance(obj, dict):
      return obj
    for k,v in obj.items():
      if not isinstance(v,dict):
        if k in {'cls','pos_emb','bias','scale'}:
          obj[k] = False
        else:
          obj[k] = True
    return obj
  def is_leaf(obj):
    if not isinstance(obj, dict):
      return True
    modify_value_based_on_key(obj)
    b = isinstance(obj, dict) and all([not isinstance(v, dict) for v in obj.values()])
    return b
  u = jax.tree_util.tree_map(lambda x:False,params)
  modified_tree = jax.tree_util.tree_map(partial(modify_value_based_on_key), u, is_leaf=is_leaf)
  return modified_tree

def create_train_state(
  config: ml_collections.ConfigDict, model, learning_rate_fn
):
  """
  Create initial training state, including the model and optimizer.
  config: the training config
  """
  # print("here we are in the function 'create_train_state' in train.py; ready to define optimizer")
  graphdef, params, batch_stats, rng_states, useless_variable_states = nn.split(model, nn.Param, nn.BatchStat, nn.RngState, nn.VariableState)

  def apply_fn(graphdef2, params2, rng_states2, batch_stats2, useless_, is_training, x, t):
    merged_model = nn.merge(graphdef2, params2, rng_states2, batch_stats2, useless_)
    if is_training:
      merged_model.train()
    else:
      merged_model.eval()
    del params2, rng_states2, batch_stats2, useless_
    out = merged_model(x, t)
    new_batch_stats, new_rng_states, _ = nn.state(merged_model, nn.BatchStat, nn.RngState, ...)
    return out, new_batch_stats, new_rng_states

  # here is the optimizer

  # if config.optimizer == 'sgd':
  #   if config.weight_decay != 0.0:
  #     print("Warning from sqa: weight decay is not supported in SGD")
  #   if config.grad_norm_clip != "None":
  #     print("Warning from sqa: grad norm clipping is not supported in SGD")
  #   tx = optax.sgd(
  #     learning_rate=learning_rate_fn,
  #     momentum=config.momentum,
  #     nesterov=True,
  #   )
  # elif config.optimizer == 'adamw':
  #   grad_norm_clip = None if config.grad_norm_clip == "None" else config.grad_norm_clip
  #   tx = optax.adamw(
  #     learning_rate=learning_rate_fn,
  #     b1=0.9,
  #     b2=0.999,
  #     eps=1e-8,
  #     weight_decay=config.weight_decay,
  #     # grad_norm_clip=grad_norm_clip, # None if no clipping
  #   )
  # else:
  #   raise ValueError(f'Unknown optimizer: {config.optimizer}, choose from "sgd" or "adamw"')
  # use adamw optimizer
  tx = optax.adamw(
    learning_rate=learning_rate_fn,
    b1=0.9,
    b2=0.95,
    eps=1e-8,
    weight_decay=config.weight_decay,
  )
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

def prepare_batch(batch, rng, training_config):
  # NOTE: there are two batch sizes: image shape is [b1, b2, 32, 32, 3].
  b1, b2 = batch['image'].shape[:2]
  sigma_m = training_config.sigma_min
  x = batch['image']
  eps = jax.random.normal(rng.train_eps(), x.shape)
  t = jax.random.uniform(rng.train_eps(), (b1, b2, 1, 1, 1))
  psi_t = (1 - (1 - sigma_m) * t) * eps + t * x
  target = x - (1 - sigma_m) * eps
  return psi_t, t.reshape(b1, b2), target

p_train_step = jax.pmap(train_step_compute,axis_name='batch')
p_solve_diffeq = jax.pmap(
  partial(solve_diffeq,see_steps=see_steps),
  axis_name='batch')

def sample_step(state:NNXTrainState, image_size, config, epoch, verbose=False, see_steps:int=10):
  '''
  config: is the sampling config
  '''
  log_for_0(f"start generating samples for epoch {epoch}")
  device_count = jax.local_device_count()
  init_x = jax.random.normal(jax.random.key(0), (device_count, 64 // device_count, image_size, image_size, 3)) # (8, 8, 32, 32, 3)
  result = p_solve_diffeq(init_x,train_state) # (8, 10, 8, 32, 32, 3)

  from input_pipeline import MEAN_RGB, STDDEV_RGB
  result = result * (jnp.array(STDDEV_RGB)/255.).reshape(1,1,1,1,1,3) + (jnp.array(MEAN_RGB)/255.).reshape(1,1,1,1,1,3)

  result = result.transpose(0,2,1,3,4,5)
  result = result.reshape((result.shape[0] * result.shape[1], *result.shape[2:])) # (num_samples, see_steps, image_size, image_size, 3)

  if jax.process_index() == 0:
    # flow_path = path.replace('.png','_flow.png')
    # gen_path = path.replace('.png','_gen.png')
    flow_dir = config.save_dir + "flow/"
    gen_dir = config.save_dir + "gen/"
    im1 = save_img(result[:9].reshape(-1,*result.shape[-3:]), flow_dir, im_name=f"{epoch}.png", grid=(10, see_steps))
    im2 = save_img(result[:,-1], gen_dir, im_name=f"{epoch}.png", grid=(8, 8))

    if use_wandb:
      wandb.log({
        'flow': wandb.Image(im1),
        'gen': wandb.Image(im2)
      })

  log_for_0(f"saved samples for epoch {epoch}")

# def denoising_eval_step(state:NNXTrainState, rng_init, sigmas, config, ground_truth, type_, epoch):

#   assert type_ in {"even", "lower"}
#   num_replicas = jax.local_device_count()
#   assert 64 % num_replicas == 0
#   local_batch_size = 64 // num_replicas

#   ground_truth_1 = ground_truth[:, :local_batch_size] # shape (num_replicas, local_bs, 28, 28, 1)
#   # print("ground_truth_1.shape: ", ground_truth_1.shape)
#   log_for_0(f"evaluating denoising for epoch {epoch}")
#   corrupted, mask = corruption(
#     ground_truth_1, 
#     type_=type_, 
#     rngs=rng_init, 
#     noise_scale=1, 
#     clamp=False
#   )
#   # print("corrupted.shape: ", corrupted.shape) # (8, 8, 28, 28, 1)
#   # print("mask.shape: ", mask.shape) # (8, 8, 28, 28, 1)
#   # denoising process
#   recovered = langevin_masked(
#     state,
#     x=corrupted,
#     sigmas=sigmas,
#     eps=config.eps,
#     T=config.T,
#     mask=mask,
#     rngs=rng_init,
#     whole_process=False,
#     clamp=False,
#     verbose=False # we will set to False later
#   )
#   dir=config.save_dir + f"denoising_{type_}/{epoch}"
#   save_img(recovered, dir, im_name=f"recovered.png", grid=(8, 8))
#   save_img(ground_truth_1, dir, im_name=f"groundtruth.png", grid=(8, 8))
#   save_img(corrupted, dir, im_name=f"corrupted.png", grid=(8, 8))

#   # calculate mse
#   mse = jnp.mean((recovered - ground_truth_1) ** 2)
#   log_for_0(f"mse for epoch {epoch} and type {type_}: {mse}")
#   log_for_0(f"saving recovering process for epoch {epoch} and type {type_}")

#   # save the whole process
#   ground_truth_0=ground_truth[0, local_batch_size:local_batch_size+10]
#   assert ground_truth_0.shape[0] == 10, "eval batch size is too small"
#   ground_truth_0 = jnp.tile(ground_truth_0.reshape(1, 10, 28, 28, 1), (num_replicas, 1, 1, 1, 1))
#   corrupted, mask = corruption(
#     ground_truth_0, 
#     type_=type_, 
#     rngs=rng_init, 
#     noise_scale=1, 
#     clamp=False
#   )
#   _, all_samples = langevin_masked(
#     state,
#     x=corrupted,
#     sigmas=sigmas,
#     eps=config.eps,
#     T=config.T,
#     mask=mask,
#     rngs=rng_init,
#     whole_process=True,
#     clamp=False,
#     verbose=False
#   )
#   dir=config.save_dir + f"denoising_{type_}_process/{epoch}"
#   g = all_samples.shape[0]
#   assert g%10 == 0
#   save_img(all_samples, dir, im_name=f"recovered.png", grid=(g//10, 10)) # this maybe reverse
#   log_for_0(f"saved denoising for epoch {epoch} and type {type_}")

#   return mse

########### Checkpointer ###########
checkpointer = ocp.StandardCheckpointer()
def _restore(ckpt_path, item, **restore_kwargs):
    return ocp.StandardCheckpointer.restore(checkpointer, ckpt_path, target=item)
setattr(checkpointer, 'restore', _restore)

def save_checkpoint(state:NNXTrainState, workdir):
  # TODO: this function currently emits lots of "background messages". Try to suppress them
  state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
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
  # TODO: FATAL: this "keep" param seems not being used. This must be fixed ASAP!

def restore_checkpoint(model_init_fn, state, workdir):
  abstract_model = nn.eval_shape(lambda: model_init_fn(rngs=nn.Rngs(0)))
  rng_states = state.rng_states
  abs_state = nn.state(abstract_model)
  _, useful_abs_state = abs_state.split(nn.RngState, ...)
  fake_state = {
      'mo_xing': useful_abs_state,
      'you_hua_qi': state.opt_state,
      'step': 0
  }
  loaded_state = checkpoints.restore_checkpoint(workdir, target=fake_state,orbax_checkpointer=checkpointer)
  merged_params = loaded_state['mo_xing']
  opt_state = loaded_state['you_hua_qi']
  step = loaded_state['step']
  params, batch_stats = merged_params.split(nn.Param, nn.BatchStat, nn.VariableState)
  return state.replace(
      params=params,
      rng_states=rng_states,
      batch_stats=batch_stats,
      opt_state=opt_state,
      step=step
  )

def _update_model_avg(model_avg, state_params, ema_decay):
  return jax.tree_util.tree_map(lambda x, y: ema_decay * x + (1 - ema_decay) * y, model_avg, state_params)
  # return model_avg

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
  training_config = config.training
  model_config = config.model 
  dataset_config = config.dataset
  sampling_config = config.sampling
  if rank == 0 and training_config.wandb:
    wandb.init(project='sqa_FM', dir=workdir)
    wandb.config.update(config.to_dict())
  global_seed(training_config.seed)

  image_size = dataset_config.image_size
  assert image_size == 28

  log_for_0('config.batch_size: {}'.format(training_config.batch_size))

  # print("save dir: ", sampling_config.save_dir)
  if sampling_config.save_dir is None:
    sampling_config.save_dir = workdir + "/images/"
  log_for_0(f"save directory: {sampling_config.save_dir}")

  ########### Create DataLoaders ###########
  if training_config.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = training_config.batch_size // jax.process_count()
  log_for_0('local_batch_size: {}'.format(local_batch_size))
  log_for_0('jax.local_device_count: {}'.format(jax.local_device_count()))

  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  train_set = train_set_(root=dataset_config.root)
  val_set = val_set_(root=dataset_config.root)

  train_loader, steps_per_epoch = create_split(
    train_set, local_batch_size, 'train', dataset_config
  )

  # eval_loader, steps_per_eval = create_split(
  #   val_set, local_batch_size, 'val', config
  # )

  eval_loader = DataLoader(val_set, batch_size=training_config.eval_batch_size, shuffle=True, drop_last=False, pin_memory=True)

  log_for_0('steps_per_epoch: {}'.format(steps_per_epoch))

  if training_config.steps_per_eval != -1:
    steps_per_eval = training_config.steps_per_eval

  ########### Create Model ###########
  # TODO: change the ncsnv2 filename
  ####### 我不会!!!!!!!!!!!!!!
  model_cls = getattr(ncsnv2, model_config.name)
  rngs = nn.Rngs(training_config.seed, params=training_config.seed + 114, dropout=training_config.seed + 514, evaluation=training_config.seed + 1919)
  dtype = get_dtype(model_config.half_precision)
  # TODO: change the API
  model_init_fn = partial(
    model_cls, 
    dtype=dtype, 
    ngf=model_config.ngf, 
    n_noise_levels=sampling_config.n_noise_levels, 
    config=config
  )
  model = model_init_fn(rngs=rngs)
  show_dict(f'number of model parameters:{count_params(model)}')
  show_dict(display_model(model))

  ########### Create LR FN ###########
  # base_lr = training_config.learning_rate * training_config.batch_size / 256.
  base_lr = training_config.learning_rate
  learning_rate_fn = create_learning_rate_fn(
      train_config=training_config,
      base_learning_rate=base_lr,
      steps_per_epoch=train_steps,
  )

  ########### Create Train State ###########
  state = create_train_state(training_config, model, learning_rate_fn)
  # restore checkpoint
  if training_config.load_from is not None:
    if not os.path.isabs(training_config.load_from):
      raise ValueError('Checkpoint path must be absolute')
    if not os.path.exists(training_config.load_from):
      raise ValueError('Checkpoint path {} does not exist'.format(training_config.load_from))
    state = restore_checkpoint(model_init_fn ,state, training_config.load_from)
    # sanity check, as in Kaiming's code
    assert state.step > 0 and state.step % steps_per_epoch == 0, ValueError('Got an invalid checkpoint with step {}'.format(state.step))
  epoch_offset = state.step // steps_per_epoch  # sanity check for resuming

  state = ju.replicate(state) # NOTE: this doesn't split the RNGs automatically, but it is an intended behavior
  model_avg = state.params
  yierbayiyiliuqi = len(train_loader.dataset) # this equals to 60000
  # print("yierbayiyiliuqi: ", yierbayiyiliuqi)

  # use pmap to parallel training
  sigmas = get_sigmas(sampling_config)
  # p_train_step = jax.pmap(
  #   functools.partial(train_step_sqa, rng_init=rngs, sigmas=sigmas),
  #   axis_name='batch',
  # )


  

  ########### Training Loop ###########
  log_for_0('Initial compilation, this might take some minutes...')

  last_model = None
  if sampling_config.get('ema_decay'):
    assert sampling_config.ema_decay > 0.0 and sampling_config.ema_decay < 1.0, 'ema_decay should be in (0, 1)'
    log_for_0('Using EMA with decay {}'.format(sampling_config.ema_decay))
    p_update_model_avg = jax.pmap(partial(_update_model_avg, ema_decay=sampling_config.ema_decay), axis_name='batch')

  for epoch in range(epoch_offset, training_config.num_epochs):
    ########### Train ###########
    timer = Timer()
    if jax.process_count() > 1:
      train_loader.sampler.set_epoch(epoch)
    log_for_0('epoch {}...'.format(epoch))
    timer.reset()
    for n_batch, batch in enumerate(train_loader):

      images = batch[0].reshape(-1, config.dataset.channels, config.dataset.image_size, config.dataset.image_size)
      step = epoch * steps_per_epoch + n_batch
      ep = step * training_config.batch_size / yierbayiyiliuqi
      # print("images.shape: ", images.shape)
      images = prepare_batch_data_sqa(images)

      # print("batch['image'].shape:", batch['image'].shape)
      # assert False

      # # here is code for us to visualize the images
      # import matplotlib.pyplot as plt
      # import numpy as np
      # import os
      # print(images.shape)

      # # save batch["image"] to ./images/{epoch}/i.png
      # rank = jax.process_index()

      # if os.path.exists(f"/kmh-nfs-us-mount/staging/sqa/images/{n_batch}/{rank}") == False:
      #   os.makedirs(f"/kmh-nfs-us-mount/staging/sqa/images/{n_batch}/{rank}")
      # if os.path.exists(f"/kmh-nfs-ssd-eu-mount/code/qiao/NCSN/sqa_NCSN/images/{n_batch}/{rank}") == False:
      #   os.makedirs(f"/kmh-nfs-ssd-eu-mount/code/qiao/NCSN/sqa_NCSN/images/{n_batch}/{rank}")
      # for i in range(images[0].shape[0]):
      #   # print the max and min of the image
      #   # print(f"max: {np.max(images[0][i])}, min: {np.min(images[0][i])}")
      #   img_test = images[0][:100]
      #   save_img(img_test, f"/kmh-nfs-ssd-eu-mount/code/qiao/NCSN/sqa_NCSN/images/{n_batch}/{rank}", im_name=f"{i}.png", grid=(10, 10))
      #   break
      #   # use the max and min to normalize the image to [0, 1]
      #   img = images[0][i]
      #   # img = (img - np.min(img)) / (np.max(img) - np.min(img))
      #   img = img.squeeze(-1)
      #   plt.imsave(f"/kmh-nfs-us-mount/staging/sqa/images/{n_batch}/{rank}/{i}.png", img, cmap='gray') # if MNIST, add cmap='gray'
      #   # if i>6: break

      # print(f"saving images for n_batch {n_batch}, done.")
      # if n_batch > 0:
      #   exit(114514)
      # continue


      # print(batch["image"].shape)

      # state, metrics = p_train_step(state, images) # here is the training step
      state, metrics = train_step_sqa(state, images, rngs, sigmas)
      
      if epoch == epoch_offset and n_batch == 0:
        log_for_0('Initial compilation completed. Reset timer.')

      if training_config.get('log_per_step'):
        if (step + 1) % training_config.log_per_step == 0:
          if index == 0:
            tang_reduce(metrics) 
            step_per_sec = training_config.log_per_step / timer.elapse_with_reset()
            loss_to_display = metrics['loss']
            if training_config.wandb:
              wandb.log({'train_ep:': ep, 
                        'train_loss': loss_to_display, 
                        # 'lr': learning_rate_fn(step), 
                        'step': step, 
                        'step_per_sec': step_per_sec})
            # log_for_0('epoch: {} step: {} loss: {}, step_per_sec: {}'.format(ep, step, loss_to_display, step_per_sec))
            log_for_0('step: {} loss: {}, step_per_sec: {}'.format(step, loss_to_display, step_per_sec))
      if sampling_config.get('ema_decay'):
        # EMA
        model_avg = p_update_model_avg(model_avg, state.params)

    ########### Save Checkpt ###########
    # we first save checkpoint, then do eval. Reasons: 1. if eval emits an error, then we still have our model; 2. avoid the program exits before the checkpointer finishes its job.
    # NOTE: when saving checkpoint, should sync batch stats first.
    state = sync_batch_stats(state)
    if (epoch + 1) % training_config.checkpoint_per_epoch == 0:
      # pass
      # if index == 0:
      save_checkpoint(state, workdir)
    if epoch == training_config.num_epochs - 1:
      state = state.replace(params=model_avg)

    ########### Eval ###########
    if (epoch + 1) % training_config.eval_per_epoch == 0:
      log_for_0('Eval epoch {}...'.format(epoch))
      # sync batch statistics across replicas
      eval_state = sync_batch_stats(state)
      eval_state = eval_state.replace(params=model_avg)
      average_metrics = MyMetrics(reduction=Avger)
      sample_step(eval_state, rngs, sigmas, sampling_config, epoch)
      for n_eval_batch, eval_batch in enumerate(eval_loader):
        images = eval_batch[0].reshape(-1, config.dataset.channels, config.dataset.image_size, config.dataset.image_size)
        ground_truth = prepare_batch_data_sqa(images) # 
        if n_eval_batch == 0:
          mse_lower = denoising_eval_step(eval_state, rngs, sigmas, sampling_config, ground_truth, "lower", epoch)
        if n_eval_batch == 1:
          mse_even = denoising_eval_step(eval_state, rngs, sigmas, sampling_config, ground_truth, "even", epoch)
          break
        # if (n_eval_batch + 1) % config.log_per_step == 0:
        #   if index == 0:
        #     log_for_0('eval: {}/{}'.format(n_eval_batch + 1, steps_per_eval))
        # eval_image = prepare_batch_data_sqa(eval_batch[0], local_batch_size)

        # metrics = p_eval_step(state, eval_batch) # here is the eval step
        # # print("metrics' labels shape:", metrics['labels'].shape)
        # assert metrics['labels'].shape[-1] == NUM_CLASSES
        # eval_metrics.append(metrics)
      # print("mse_lower: ", mse_lower)
      mse_lower = jnp.mean(mse_lower)
      mse_even = jnp.mean(mse_even)

      if index == 0 and training_config.wandb:
        wandb.log({'mse_lower': mse_lower, 'mse_even': mse_even, 'epoch': epoch})
        log_for_0('epoch: {}; mse_lower: {}, mse_even: {}'.format(epoch, mse_lower, mse_even))


  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  checkpointer.close() # avoid exiting before checkpt is saved
  if index == 0 and training_config.wandb:
    wandb.finish()

  return state

def just_evaluate(
    config: ml_collections.ConfigDict, workdir: str
  ):
  ########### Initialize ###########
  rank = index = jax.process_index()
  log_for_0('Generating samples for workdir: {}'.format(workdir))
  training_config = config.training
  model_config = config.model 
  dataset_config = config.dataset
  sampling_config = config.sampling
  dtype = jnp.bfloat16 if model_config.half_precision else jnp.float32
  global_seed(training_config.seed)
  sigmas = get_sigmas(sampling_config)
  image_size = dataset_config.image_size

  ########### Create Model ###########
  model_cls = getattr(ncsnv2, model_config.name)
  rngs = nn.Rngs(training_config.seed, params=training_config.seed + 114, dropout=training_config.seed + 514, evaluation=training_config.seed + 1919)
  dtype = get_dtype(model_config.half_precision)
  model_init_fn = partial(
    model_cls, 
    dtype=dtype, 
    ngf=model_config.ngf, 
    n_noise_levels=sampling_config.n_noise_levels, 
    config=config
  )
  model = model_init_fn(rngs=rngs)
  display_model(model)

  ########### Create LR FN ###########
  # base_lr = training_config.learning_rate * training_config.batch_size / 256.
  learning_rate_fn = training_config.learning_rate

  ########### Create Train State ###########
  state = create_train_state(training_config, model, learning_rate_fn)
  assert training_config.get('load_from',None) is not None, 'Must provide a checkpoint path for evaluation'
  if not os.path.isabs(training_config.load_from):
    raise ValueError('Checkpoint path must be absolute')
  if not os.path.exists(training_config.load_from):
    raise ValueError('Checkpoint path {} does not exist'.format(training_config.load_from))
  state = restore_checkpoint(model_init_fn ,state, training_config.load_from)
  state_step = int(state.step)
  state = ju.replicate(state) # NOTE: this doesn't split the RNGs automatically, but it is an intended behavior

  ########### Gen ###########
  log_for_0('Eval...')
  # sync batch statistics across replicas
  eval_state = sync_batch_stats(state)
  average_metrics = MyMetrics(reduction=Avger)
  sample_step(eval_state, rngs, sigmas, sampling_config, epoch="eval", verbose=True)

  jax.random.normal(jax.random.key(0), ()).block_until_ready()