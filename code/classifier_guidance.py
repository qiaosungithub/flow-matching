raise NotImplementedError('NFE for classifier guidance is not implemented')
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

class P:
  def 不高兴的(self):
    return False
你 = P()

import models.models_ddpm as models_ddpm
import models.models_ddpm_classifier as models_ddpm_classifier
from models.models_ddpm import generate, edm_ema_scales_schedules, diffusion_schedule_fn_some, create_zhh_SAMPLING_diffusion_schedule

NUM_CLASSES = 10

class NNXTrainState(FlaxTrainState):
  batch_stats: Any
  rng_states: Any
  graphdef: Any
  useless_variable_state: Any
  # NOTE: is_training can't be a attr, since it can't be replicated


def sample_step(state, sample_idx, model, rng_init, device_batch_size, config,zhh_o,MEAN_RGB=None, STDDEV_RGB=None,option='FID',classifier=None,classifier_state=None):
  """
  sample_idx: each random sampled image corrresponds to a seed
  rng_init: here we do not want nnx.Rngs
  """
  rng_sample = random.fold_in(rng_init, sample_idx)  # fold in sample_idx
  # images, denoised = generate(state, model, rng_sample, n_sample=device_batch_size, config=config,zhh_o=zhh_o) # for debug
  images, nfe = generate(state, model, rng_sample, n_sample=device_batch_size, config=config,zhh_o=zhh_o,label_type=('order' if option=='vis' else 'random'),classifier=classifier,classifier_state=classifier_state,classifier_scale=config.classifier_scale) # we force conditional generation here

  images_all = lax.all_gather(images, axis_name='batch')  # each device has a copy  
  images_all = images_all.reshape(-1, *images_all.shape[2:])
  
  
  # debug
  # denoised_all = lax.all_gather(denoised, axis_name='batch')  # each device has a copy
  # denoised_all = denoised_all.reshape(-1, *denoised_all.shape[2:])

  # The images should be [-1, 1], which is correct

  # images_all = images_all * (jnp.array(STDDEV_RGB)/255.).reshape(1,1,1,3) + (jnp.array(MEAN_RGB)/255.).reshape(1,1,1,3)
  # images_all = (images_all - 0.5) / 0.5
  # return images_all, denoised_all # debug
  return images_all, nfe

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

def restore_checkpoint(model_init_fn, state, workdir, model_config, ema=False,is_classifier=False):
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
    # 'ema_mo_xing': params_1,
    # 'ema_mo_xing': useful_abs_state_1,
    'you_hua_qi': state.opt_state,
    'step': 0
  }
  if not is_classifier:
    fake_state['ema_mo_xing'] = useful_abs_state
  else:
    assert not ema, 'Classifier does not support EMA'
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
  ), loaded_state.get('ema_mo_xing', None) if not ema else loaded_state['mo_xing']

# zhh's nnx version

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


def train_and_evaluate(
  config: ml_collections.ConfigDict, workdir: str
) -> NNXTrainState:
  raise NotImplementedError('classifier guidance only do inference')

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
    wandb.init(project='LMCI-eval', dir=workdir, tags=['ADM'] if not model_config.class_conditional else ['ADM', 'Conditional'])
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
  model_init_fn = partial(model_cls, num_classes=NUM_CLASSES, dtype=dtype,**config.diffusion_schedule)
  model = model_init_fn(rngs=rngs, **model_config)
  show_dict(f'number of model parameters:{count_params(model)}')
  # show_dict(display_model(model))

  ########### Create LR FN ###########
  learning_rate_fn = lambda:114514.1919810 # just in order to create the state

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
  
  print('Model loaded successfully!')
  
  ########### Create Classifier State ###########
  classifier_cls = models_ddpm_classifier.SimDDPM  
  classifier_rngs = nn.Rngs(config.seed, params=config.seed + 810, dropout=config.seed + 666, train=config.seed + 888)
  classifier_init_fn = partial(classifier_cls, num_classes=NUM_CLASSES, dtype=dtype)
  classifier = classifier_init_fn(rngs=classifier_rngs, **config.classifier_model)
  
  classifier_state = create_train_state(config, classifier, image_size, learning_rate_fn)
  assert config.get('load_classifier_from',None) is not None, 'Must provide a checkpoint path for evaluation'
  if not os.path.isabs(config.load_classifier_from):
    raise ValueError('Checkpoint path must be absolute')
  if not os.path.exists(config.load_classifier_from):
    raise ValueError('Checkpoint path {} does not exist'.format(config.load_classifier_from))
  classifier_state,_ = restore_checkpoint(classifier_init_fn, classifier_state, config.load_classifier_from, config.classifier_model, ema=False, is_classifier=True) # NOTE: whether to use the ema model
  
  print('Classifier loaded successfully!')
  
  ########### FID ###########
  vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())  # for visualization
  if config.model.ode_solver == 'jax':
    p_sample_step = jax.pmap(
      partial(sample_step, 
              model=model, 
              classifier=classifier,
              classifier_state=classifier_state,
              rng_init=random.PRNGKey(0), 
              device_batch_size=config.fid.device_batch_size, 
              config=config,
              zhh_o = create_zhh_SAMPLING_diffusion_schedule(config),
              option='FID'
              # MEAN_RGB=input_pipeline.MEAN_RGB, 
              # STDDEV_RGB=input_pipeline.STDDEV_RGB
      ),
      axis_name='batch'
    )
    p_visualize_sample_step = jax.pmap(
      partial(sample_step, 
              model=model, 
              classifier=classifier,
              classifier_state=classifier_state,
              rng_init=random.PRNGKey(0), 
              device_batch_size=100, 
              config=config,
              zhh_o = create_zhh_SAMPLING_diffusion_schedule(config),
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
      # images, denoised = p_sample_step(state, sample_idx=sample_idx) # debug
      images = p_sample_step_(state, sample_idx=sample_idx)
      # print("In function run_p_sample_step; images.shape: ", images.shape, flush=True)
      jax.random.normal(random.key(0), ()).block_until_ready()
      # return images[0], denoised[0]  # images have been all /gathered
      return images[0]  # images have been all gathered
    
  elif config.model.ode_solver == 'scipy':
    from utils.rk45_util import get_rk45_functions
    run_p_sample_step, p_sample_step = get_rk45_functions(model, config, random.PRNGKey(0))

  else:
    raise NotImplementedError('Unsupported ode_solver: {}'.format(config.model.ode_solver))
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
    vis = run_p_sample_step(p_visualize_sample_step, eval_state, vis_sample_idx)
    # assert False, vis.shape
    vis = make_grid_visualization(vis,grid=10,max_bz=10)
    vis = jax.device_get(vis) # np.ndarray
    vis = vis[0]
    # print(vis.shape)
    # exit("王广廷")
    canvas = Image.fromarray(vis)
    # canvas.save(f"{workdir}/sample.png")
    # assert False, 'Image saved'
    
    # vis, denoise_vis = run_p_sample_step(p_sample_step, eval_state, vis_sample_idx)
    # print('saving images...')
    # for step,onevis in enumerate(vis):
    #   onevis = make_grid_visualization(onevis)
    #   onevis = jax.device_get(onevis) # np.ndarray
    #   onevis = onevis[0]
    # # print(vis.shape)
    # # exit("王广廷")
    #   canvas = Image.fromarray(onevis)
    #   # save the image
    #   canvas.save(f"{workdir}/{step:03d}_sample.png")
      
    # for step,onevis in enumerate(denoise_vis):
    #   onevis = make_grid_visualization(onevis)
    #   onevis = jax.device_get(onevis)
    #   onevis = onevis[0]
    #   canvas = Image.fromarray(onevis)
    #   canvas.save(f"{workdir}/{step:03d}_denoise.png")
    
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

  if rank == 0 and config.wandb:
    nfe = config.model.n_T
    if config.model.ode_solver == 'scipy': nfe=100
    elif config.model.sampler not in ['euler', 'ddpm']: nfe*=2
    wandb.log({'NFE': nfe})

  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  if index == 0 and config.wandb:
    wandb.finish()