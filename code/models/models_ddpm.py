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

"""Flax implementation of ResNet V1.5."""

# See issue #620.
# pytype: disable=wrong-arg-count

from absl import logging
from typing import Any, Sequence

# from flax import linen as nn
import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import math
from flax.training.train_state import TrainState as FlaxTrainState

from functools import partial

# from models.models_unet import ContextUnet
from models.models_ncsnpp_edm import NCSNpp as NCSNppEDM
# from models.models_ncsnpp import NCSNpp
# import models.jcm.sde_lib as sde_lib
from models.jcm.sde_lib import batch_mul



ModuleDef = Any


# these are copied from IDDPM repo

### cosine schedule
        # return betas_for_alpha_bar(
        #     num_diffusion_timesteps,
        #     lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        # )
        
### Gaussian Diffusion

# cache diffusion schedule
zhh_diffusion_schedule = None

class NNXTrainState(FlaxTrainState):
  batch_stats: Any
  rng_states: Any
  graphdef: Any
  useless_variable_state: Any
  # NOTE: is_training can't be a attr, since it can't be replicated


def ct_ema_scales_schedules(step, config, steps_per_epoch):
  start_ema = float(config.ct.start_ema)
  start_scales = int(config.ct.start_scales)
  end_scales = int(config.ct.end_scales)
  total_steps = config.num_epochs * steps_per_epoch

  scales = jnp.ceil(jnp.sqrt((step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2) + start_scales**2) - 1).astype(jnp.int32)
  scales = jnp.maximum(scales, 1)
  c = -jnp.log(start_ema) * start_scales
  target_ema = jnp.exp(-c / scales)
  scales = scales + 1
  return target_ema, scales


def edm_ema_scales_schedules(step, config, steps_per_epoch):
  # ema_halflife_kimg = 500  # from edm
  ema_halflife_kimg = 50000  # log(0.5) / log(0.999999) * 128 / 1000 = 88722 kimg, from flow
  ema_halflife_nimg = ema_halflife_kimg * 1000

  ema_rampup_ratio = 0.05
  ema_halflife_nimg = jnp.minimum(ema_halflife_nimg, step * config.batch_size * ema_rampup_ratio)

  ema_beta = 0.5 ** (config.batch_size / jnp.maximum(ema_halflife_nimg, 1e-8))
  scales = jnp.ones((1,), dtype=jnp.int32)
  return ema_beta, scales

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return jnp.array(betas)
  
def create_zhh_diffusion_schedule(config):
  global zhh_diffusion_schedule
  if zhh_diffusion_schedule is None:
    if config.diffusion_schedule == 'cosine':
      betas =  betas_for_alpha_bar(
              config.diffusion_nT,
              lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
          )
      # 大便
      alphas = 1.0 - betas
      alphas_cumprod = jnp.cumprod(alphas, axis=0)
      alphas_cumprod_prev = jnp.append(1.0, alphas_cumprod[:-1])
      alphas_cumprod_next = jnp.append(alphas_cumprod[1:], 0.0)
      global zhh_diffusion_schedule
      zhh_diffusion_schedule = {
        'betas': betas, 'alphas': alphas, 'alpha_cumprod': alphas_cumprod, 'alpha_cumprod_prev': alphas_cumprod_prev, 'alpha_cumprod_next': alphas_cumprod_next
      }
      # cosine schedule
    elif config.diffusion_schedule == 'linear':
      raise NotImplementedError
    else:
      raise NotImplementedError
  return zhh_diffusion_schedule

def create_zhh_SAMPLING_diffusion_schedule(config):
    diffusion_steps_total = config.diffusion_nT
    sample_steps_total = config.model.nT
    assert sample_steps_total == diffusion_steps_total, 'This is not supported: sample_steps_total {} != diffusion_steps_total {}'.format(sample_steps_total, diffusion_steps_total)
    o = create_zhh_diffusion_schedule(config)
    if 'sample_ts' not in o:
        sample_ts = jnp.flip(jnp.arange(diffusion_steps_total), axis=0)
        # 更多的大便
        posterior_variance = o['betas'] * (1.0 - o['alphas_cumprod_prev']) / (1.0 - o['alphas_cumprod'])
        model_variance = jnp.append(posterior_variance[1],o['betas'][1:])
        log_model_variance = jnp.log(model_variance)
        sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / o['alphas_cumprod'])
        sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / o['alphas_cumprod'] - 1.0)
        posterior_mean_coef1 = (
            o['betas'] * jnp.sqrt(o['alphas_cumprod_prev']) / (1.0 - o['alphas_cumprod'])
        )
        posterior_mean_coef2 = (
            (1.0 - o['alphas_cumprod_prev'])
            * np.sqrt(o['alphas'])
            / (1.0 - o['alphas_cumprod'])
        )
        global zhh_diffusion_schedule
        zhh_diffusion_schedule.update({
            'sample_ts': sample_ts,
            'posterior_variance': posterior_variance,
            'model_variance': model_variance,
            'log_model_variance': log_model_variance,
            'sqrt_recip_alphas_cumprod': sqrt_recip_alphas_cumprod,
            'sqrt_recipm1_alphas_cumprod': sqrt_recipm1_alphas_cumprod,
            'posterior_mean_coef1': posterior_mean_coef1,
            'posterior_mean_coef2': posterior_mean_coef2
        })
    global zhh_diffusion_schedule
    return zhh_diffusion_schedule
    

def diffusion_schedule_fn_some(config, t_batch):
  o = create_zhh_diffusion_schedule(config)
  return o['alpha_cumprod'][t_batch], o['beta'][t_batch]

def get_t_process_fn(t_condition_method):
  if t_condition_method == 'log999':
    return lambda t: jnp.log(t * 999)
  elif t_condition_method == 'direct':
    return lambda t: t
  elif t_condition_method == 'not': # no t: 没有t
    return lambda t: jnp.zeros_like(t)
  else:
    raise NotImplementedError('Unknown t_condition_method: {m}'.format(m=t_condition_method))

# move this out from model for JAX compilation
def generate(state: NNXTrainState, model, rng, n_sample,config):
  """
  Generate samples from the model

  Here we tend to not use nnx.Rngs
  state: maybe a train state
  ---
  return shape: (n_sample, 32, 32, 3)
  """

  # prepare schedule
  num_steps = model.n_T

  # initialize noise
  x_shape = (n_sample, model.image_size, model.image_size, model.out_channels)
  rng_used, rng = jax.random.split(rng, 2)
  # sample from prior
  x_prior = jax.random.normal(rng_used, x_shape, dtype=model.dtype)


  if model.sampler in ['euler', 'heun']:
      
    x_i = x_prior

    def step_fn(i, inputs):
      x_i, rng = inputs
      rng_this_step = jax.random.fold_in(rng, i)
      rng_z, 别传进去 = jax.random.split(rng_this_step, 2)

      merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
      x_i = merged_model.sample_one_step(x_i, rng_z, i)
      outputs = (x_i, rng)
      return outputs

    outputs = jax.lax.fori_loop(0, num_steps, step_fn, (x_i, rng))
    images = outputs[0]
    return images
  
  elif model.sampler in ['edm', 'edm-sde']:
    t_steps = model.compute_t(jnp.arange(num_steps), num_steps)
    t_steps = jnp.concatenate([t_steps, jnp.zeros((1,), dtype=model.dtype)], axis=0)  # t_N = 0; no need to round_sigma
    x_i = x_prior * t_steps[0]

    # import jax.random as random
    # x = random.normal(rng, x_shape, dtype=model.dtype)

    def step_fn(i, inputs):
      x_i, rng = inputs
      rng_this_step = jax.random.fold_in(rng, i)
      rng_z, 别传进去 = jax.random.split(rng_this_step, 2)

      merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
      x_i = merged_model.sample_one_step_edm(x_i, rng_z, i, t_steps)
      # x_i, denoised = merged_model.sample_one_step_edm(x_i, rng_z, i, t_steps) # for debug

      outputs = (x_i, rng)
      return outputs
      # return outputs, denoised # for debug

    outputs = jax.lax.fori_loop(0, num_steps, step_fn, (x_i, rng))
    images = outputs[0]
    return images
    # # for debug
    # all_x = []
    # denoised = []
    # for i in range(num_steps):
    #   D = step_fn(i, (x_i, rng))
    #   x_i, rng = D[0]
    #   denoised.append(D[1])
    #   all_x.append(x_i)
    # images = jnp.stack(all_x, axis=0)
    # denoised = jnp.stack(denoised, axis=0)
    # return images, denoised
  elif model.sampler in ['ddpm']:
    x_i = x_prior
    o = create_zhh_SAMPLING_diffusion_schedule(config)
    t_steps = o['sample_ts']
    sqrt_recip_alphas_cumprod_steps = o['sqrt_recip_alphas_cumprod']
    sqrt_recipm1_alphas_cumprod_steps = o['sqrt_recipm1_alphas_cumprod']
    posterior_mean_coef1_steps = o['posterior_mean_coef1']
    posterior_mean_coef2_steps = o['posterior_mean_coef2']
    log_model_variance_steps = o['log_model_variance']

    def step_fn(i, inputs):
      x_i, rng = inputs
      rng_this_step = jax.random.fold_in(rng, i)
      rng_z, 别传进去 = jax.random.split(rng_this_step, 2)

      merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
      x_i = merged_model.sample_one_step_ddpm(merged_model, x_i, rng_z, i, t_steps=t_steps, sqrt_recip_alphas_cumprod_steps=sqrt_recip_alphas_cumprod_steps, sqrt_recipm1_alphas_cumprod_steps=sqrt_recipm1_alphas_cumprod_steps,posterior_mean_coef1_steps=posterior_mean_coef1_steps,posterior_mean_coef2_steps=posterior_mean_coef2_steps,
      log_model_variance_steps=log_model_variance_steps)
      outputs = (x_i, rng)
      return outputs

    outputs = jax.lax.fori_loop(0, num_steps, step_fn, (x_i, rng))
    images = outputs[0]
    return images
  
  else:
    raise NotImplementedError

class SimDDPM(nn.Module):
  """Simple DDPM."""

  def __init__(self,
    image_size,
    base_width,
    num_classes = 10,
    out_channels = 1,
    P_std = 1.2,
    P_mean = -1.2,
    n_T = 18,  # inference steps
    net_type = 'ncsnpp',
    dropout = 0.0,
    dtype = jnp.float32,
    use_aug_label = False,
    average_loss = False,
    eps=1e-3,
    h_init=0.035,
    sampler='euler',
    sample_clip_denoised=True,
    ode_solver='jax',
    no_condition_t=False,
    t_condition_method = 'log999',
    rngs=None,
    **kwargs
  ):
    self.image_size = image_size
    self.base_width = base_width
    self.num_classes = num_classes
    self.out_channels = out_channels
    self.P_std = P_std
    self.P_mean = P_mean
    self.n_T = n_T
    self.net_type = net_type
    self.dropout = dropout
    self.dtype = dtype
    self.use_aug_label = use_aug_label
    self.average_loss = average_loss
    self.eps = eps
    self.h_init = h_init
    self.sampler = sampler
    self.ode_solver = ode_solver
    # self.no_condition_t = no_condition_t
    assert no_condition_t == False, 'This is deprecated'
    self.t_preprocess_fn = get_t_process_fn(t_condition_method)
    self.rngs = rngs
    self.sample_clip_denoised = sample_clip_denoised

    # sde = sde_lib.KVESDE(
    #   t_min=0.002,
    #   t_max=80.0,
    #   N=18,  # config.model.num_scales
    #   rho=7.0,
    #   data_std=0.5,
    # )
    # self.sde = sde
    # This is not used in flow matching

    if self.net_type == 'context':
      raise NotImplementedError
      net_fn = partial(ContextUnet,
        in_channels=self.out_channels,
        n_feat=self.base_width,
        n_classes=self.num_classes,
        image_size=self.image_size,)
    elif self.net_type == 'ncsnpp':
      raise NotImplementedError
      net_fn = partial(NCSNpp,
        base_width=self.base_width,
        image_size=self.image_size,
        dropout=self.dropout)
    elif self.net_type == 'ncsnppedm':
      net_fn = partial(NCSNppEDM,
        base_width=self.base_width,
        image_size=self.image_size,
        out_channels=self.out_channels,
        dropout=self.dropout,
        rngs=self.rngs)
    else:
      raise ValueError(f'Unknown net type: {self.net_type}')

    # # declare two networks
    # self.net = net_fn(name='net')
    # self.net_ema = net_fn(name='net_ema')
    self.net = net_fn()


  def get_visualization(self, list_imgs):
    vis = jnp.concatenate(list_imgs, axis=1)
    return vis

  def compute_t(self, indices, scales):
    t_max = 80
    t_min = 0.002
    rho = 7.0
    t = t_max ** (1 / rho) + indices / (scales - 1) * (
        t_min ** (1 / rho) - t_max ** (1 / rho)
    )
    t = t**rho
    return t
    
  def compute_losses(self, pred, gt):
    assert pred.shape == gt.shape

    # simple l2 loss
    loss_rec = jnp.mean((pred - gt)**2)
    
    loss_train = loss_rec

    dict_losses = {
      'loss_rec': loss_rec,
      'loss_train': loss_train
    }
    return loss_train, dict_losses

  def sample_one_step(self, x_i, rng, i):

    if self.sampler == 'euler':
      x_next = self.sample_one_step_euler(x_i, i) 
    elif self.sampler == 'heun':
      x_next = self.sample_one_step_heun(x_i, i) 
    elif self.sampler == 'edm':
      raise LookupError("找对地方了")
    else:
      raise NotImplementedError

    return x_next
  
  def sample_one_step_edm(self, x_i, rng, i, t_steps):

    if self.sampler == 'edm':
      x_next = self.sample_one_step_edm_ode(x_i, i, t_steps) 
      # x_next, denoised = self.sample_one_step_edm_ode(x_i, i, t_steps) # for debug
    elif self.sampler == 'edm-sde':
      x_next = self.sample_one_step_edm_sde(x_i, rng, i, t_steps)
      # x_next, denoised = self.sample_one_step_edm_sde(x_i, rng, i, t_steps) # for debug
    else:
      raise NotImplementedError

    return x_next
    # return x_next, denoised 

  def sample_one_step_heun(self, x_i, i):

    x_cur = x_i

    t_cur = i / self.n_T  # t start from 0 (t = 0 is noise here)
    t_cur = t_cur * (1 - self.eps) + self.eps

    t_next = (i + 1) / self.n_T  # t start from 0 (t = 0 is noise here)
    t_next = t_next * (1 - self.eps) + self.eps

    # TODO{kaiming}: revisit S_churn
    t_hat = t_cur
    x_hat = x_cur  # x_hat is always x_cur when gamma=0

    t_hat = jnp.repeat(t_hat, x_hat.shape[0])
    t_next = jnp.repeat(t_next, x_hat.shape[0])
    
    # Euler step.
    u_pred = self.forward_flow_pred_function(x_i, t_hat, train=False)
    d_cur = u_pred
    x_next = x_hat + batch_mul(u_pred, t_next - t_hat)

    # Apply 2nd order correction
    u_pred = self.forward_flow_pred_function(x_next, t_next, train=False)
    d_prime = u_pred
    x_next_ = x_hat + batch_mul(0.5 * d_cur + 0.5 * d_prime, t_next - t_hat)

    x_next = jnp.where(i < self.n_T - 1, x_next_, x_next)

    return x_next

  def sample_one_step_euler(self, x_i, i):
    # i: loop from 0 to self.n_T - 1
    t = i / self.n_T  # t start from 0 (t = 0 is noise here)
    t = t * (1 - self.eps) + self.eps
    t = jnp.repeat(t, x_i.shape[0])

    u_pred = self.forward_flow_pred_function(x_i, t, train=False)

    # move one step
    dt = 1. / self.n_T
    x_next = x_i + u_pred * dt

    return x_next
  
  def sample_one_step_edm_ode(self, x_i, i, t_steps):
    """
    edm's second order ODE solver
    """

    x_cur = x_i
    t_cur = t_steps[i]
    t_next = t_steps[i + 1]

    # TODO{kaiming}: revisit S_churn
    t_hat = t_cur
    x_hat = x_cur  # x_hat is always x_cur when gamma=0

    t_hat = jnp.repeat(t_hat, x_hat.shape[0])
    t_next = jnp.repeat(t_next, x_hat.shape[0])
    
    # Euler step.
    denoised = self.forward_edm_denoising_function(x_hat, t_hat, train=False)
    d_cur = batch_mul(x_hat - denoised, 1. / t_hat)
    x_next = x_hat + batch_mul(d_cur, t_next - t_hat)

    # Apply 2nd order correction
    denoised = self.forward_edm_denoising_function(x_next, t_next, train=False)
    d_prime = batch_mul(x_next - denoised, 1. / jnp.maximum(t_next, 1e-8))  # won't take effect if t_next is 0 (last step)
    x_next_ = x_hat + batch_mul(0.5 * d_cur + 0.5 * d_prime, t_next - t_hat)

    x_next = jnp.where(i < self.n_T - 1, x_next_, x_next)

    # return x_next, denoised # for debug
    return x_next
  
  def sample_one_step_edm_sde(self, x_i, rng, i, t_steps):
    """
    edm's second order SDE solver
    """

    gamma = jnp.minimum(30/self.n_T, jnp.sqrt(2)-1)
    # gamma = jnp.minimum(80/self.n_T, jnp.sqrt(2)-1)
    S_noise = 1.007
    t_max = 1
    t_min = 0.01
    # t_min = 0.05

    x_cur = x_i
    t_cur = t_steps[i]
    t_next = t_steps[i + 1]

    # jax.debug.print('t_cur shape: {s}', s=t_cur.shape)
    # jax.debug.print('i shape: {s}', s=i.shape)
    # jax.debug.print('t_steps shape: {s}', s=t_steps.shape)

    gamma = jnp.where(t_cur < t_max, gamma, 0)
    gamma = jnp.where(t_cur > t_min, gamma, 0)

    t_hat = t_cur * (1 + gamma)
    x_hat = x_cur + jnp.sqrt(t_hat**2 - t_cur**2) * S_noise * jax.random.normal(rng, x_cur.shape) # add noise to t_hat level

    t_hat = jnp.repeat(t_hat, x_hat.shape[0])
    t_next = jnp.repeat(t_next, x_hat.shape[0])
    
    # Euler step.
    denoised = self.forward_edm_denoising_function(x_hat, t_hat, train=False)
    d_cur = batch_mul(x_hat - denoised, 1. / t_hat)
    x_next = x_hat + batch_mul(d_cur, t_next - t_hat)

    # Apply 2nd order correction
    denoised = self.forward_edm_denoising_function(x_next, t_next, train=False)
    d_prime = batch_mul(x_next - denoised, 1. / jnp.maximum(t_next, 1e-8))  # won't take effect if t_next is 0 (last step)
    x_next_ = x_hat + batch_mul(0.5 * d_cur + 0.5 * d_prime, t_next - t_hat)

    x_next = jnp.where(i < self.n_T - 1, x_next_, x_next)

    # return x_next, denoised # for debug
    return x_next

  def sample_one_step_ddpm(self, x_i, rng, i, t_steps, sqrt_recip_alphas_cumprod_steps, sqrt_recipm1_alphas_cumprod_steps, posterior_mean_coef1_steps, posterior_mean_coef2_steps,log_model_variance_steps):
      """
      DDPM
      """
      t = t_steps[i]
      sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod_steps[i]
      sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod_steps[i]
      posterior_mean_coef1 = posterior_mean_coef1_steps[i]
      posterior_mean_coef2 = posterior_mean_coef2_steps[i]
      log_model_variance = log_model_variance_steps[i]
      
      eps_pred = self.forward_flow_pred_function(x_i, t, train=False)
      
      
      # get x_start from eps
      x_start = batch_mul(sqrt_recip_alphas_cumprod, x_i) - batch_mul(sqrt_recipm1_alphas_cumprod, eps_pred)
      
      if self.sample_clip_denoised:
          x_start = jnp.clip(x_start, -1, 1)
          
      model_mean = batch_mul(posterior_mean_coef1, x_start) + batch_mul(posterior_mean_coef2, x_i) # 这里写错了就高兴
      
      
      noise = jax.random.normal(rng, x_i.shape)

      nonzero_mask = jnp.where(t > 0.5, 1, 0) # no noise when t == 0
      sample = model_mean + batch_mul(nonzero_mask, noise) * jnp.exp(0.5 * log_model_variance)

      # return x_next, denoised # for debug
      return sample


  def forward_consistency_function(self, x, t, pred_t=None):
    raise NotImplementedError
    c_in = 1 / jnp.sqrt(t**2 + self.sde.data_std**2)
    in_x = batch_mul(x, c_in)  # input scaling of edm
    cond_t = 0.25 * jnp.log(t)  # noise cond of edm

    # forward
    denoiser = self.net(in_x, cond_t)

    if pred_t is None:  # TODO: what's this?
      pred_t = self.sde.t_min

    c_out = (t - pred_t) * self.sde.data_std / jnp.sqrt(t**2 + self.sde.data_std**2)
    denoiser = batch_mul(denoiser, c_out)

    c_skip = self.sde.data_std**2 / ((t - pred_t) ** 2 + self.sde.data_std**2)
    skip_x = batch_mul(x, c_skip)

    denoiser = skip_x + denoiser

    return denoiser

  def forward_flow_pred_function(self, z, t, augment_label=None, train: bool = True):  # EDM

    # t_cond = jnp.zeros_like(t) if self.no_condition_t else jnp.log(t * 999)
    t_cond = self.t_preprocess_fn(t).astype(self.dtype)
    u_pred = self.net(z, t_cond, augment_label=augment_label, train=train)
    return u_pred
  
  def forward_edm_denoising_function(self, x, sigma, augment_label=None, train: bool = True):  # EDM
    """
    code from edm
    ---
    input: x (noisy image, =x+sigma*noise), sigma (condition)
    We hope this function operates D(x+sigma*noise) = x
    our network has F((1-t)x + t*noise) = x - noise
    """

    # forward network
    c_in = 1 / (sigma + 1)
    in_x = batch_mul(x, c_in)
    c_out = sigma / (sigma + 1)

    F_x = self.forward_flow_pred_function(in_x, c_in, augment_label=augment_label, train=train)

    D_x = in_x + batch_mul(F_x, c_out)
    return D_x

  def forward(self, imgs, labels, augment_label, noise_batch, t_batch,alpha_cumprod_batch, beta_batch, train: bool = True):
    """
    You should first sample the noise and t and input them
    """
    
    # TODO: write here
    
    imgs = imgs.astype(self.dtype)
    gt = imgs
    x = imgs
    bz = imgs.shape[0]

    assert noise_batch.shape == x.shape
    assert t_batch.shape == (bz,)
    # t_batch = t_batch.reshape(bz, 1, 1, 1)

    # -----------------------------------------------------------------
    #  diffusion alpha, betas
    betas = beta_batch
    assert betas.shape == t_batch.shape, 'betas shape: {s}, t_batch shape: {t}'.format(s=betas.shape, t=t_batch.shape)
    alphas = 1.0 - betas
    alphas_cumprod = alpha_cumprod_batch
    assert alpha_cumprod_batch.shape == t_batch.shape, 'alpha_cumprod_batch shape: {s}, t_batch shape: {t}'.format(s=alpha_cumprod_batch.shape, t=t_batch.shape)
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    
    
    
    # -----------------------------------------------------------------

    # sample from data
    x_data = x
    # sample from prior (noise)
    x_prior = noise_batch

    z = batch_mul(sqrt_alphas_cumprod, x_data) + batch_mul(sqrt_one_minus_alphas_cumprod, x_prior)
    
    t = t_batch

    # create v target
    v_target = x_prior
    # v_target = jnp.ones_like(x_data)  # dummy


    # forward network
    u_pred = self.forward_flow_pred_function(z, t)


    # loss
    loss = (v_target - u_pred)**2
    if self.average_loss:
      loss = jnp.mean(loss, axis=(1, 2, 3))  # mean over pixels
    else:
      loss = jnp.sum(loss, axis=(1, 2, 3))  # sum over pixels
    loss = loss.mean()  # mean over batch

    loss_train = loss

    dict_losses = {}
    dict_losses['loss'] = loss  # legacy
    dict_losses['loss_train'] = loss_train

    # prepare some visualization
    # if we can pred u, then we can reconstruct x_data from x_prior
    # x_data_pred = z + batch_mul(t, u_pred)
    x_data_pred = z + batch_mul(1 - t, u_pred)
    

    images = self.get_visualization(
      [gt,
       v_target,  # target of network (known)
       u_pred,
       z,  # input to network (known)
       x_data_pred,
      ])

    return loss_train, dict_losses, images

  def __call__(self, imgs, labels, train: bool = False):
    # initialization only
    t = jnp.ones((imgs.shape[0],))
    augment_label = jnp.ones((imgs.shape[0], 9)) if self.use_aug_label else None  # fixed augment_dim # TODO: what is this?
    out = self.net(imgs, t, augment_label) # TODO: whether to add train=train
    out_ema = None   # no need to initialize it here
    return out, out_ema
