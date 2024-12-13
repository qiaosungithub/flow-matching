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
from flax.training.train_state import TrainState as FlaxTrainState

from functools import partial

# from models.models_unet import ContextUnet
from models.models_ncsnpp_edm import NCSNpp as NCSNppEDM
# from models.models_ncsnpp import NCSNpp
# import models.jcm.sde_lib as sde_lib
from models.jcm.sde_lib import batch_mul



ModuleDef = Any

# def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
#     """
#     DDIM util function
#     """
#     def sigmoid(x):
#         return 1 / (jnp.exp(-x) + 1)

#     if beta_schedule == "quad":
#         betas = (
#             jnp.linspace(
#                 beta_start ** 0.5,
#                 beta_end ** 0.5,
#                 num_diffusion_timesteps,
#                 dtype=np.float64,
#             )
#             ** 2
#         )
#     elif beta_schedule == "linear":
#         betas = jnp.linspace(
#             beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
#         )
#     elif beta_schedule == "const":
#         betas = beta_end * jnp.ones(num_diffusion_timesteps, dtype=np.float64)
#     elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
#         betas = 1.0 / jnp.linspace(
#             num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
#         )
#     elif beta_schedule == "sigmoid":
#         betas = jnp.linspace(-6, 6, num_diffusion_timesteps)
#         betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
#     else:
#         raise NotImplementedError(beta_schedule)
#     assert betas.shape == (num_diffusion_timesteps,)
#     return betas


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

# from jax.experimental import ode as O
# import models.ode_pkg_repo as O
import models.ode_pkg as O
def solve_diffeq_by_O(init_x,state,see_steps:int=10,t_min:float=0.0):
    def f(x, t):
        # assert t.shape == (), ValueError(f't shape: {t.shape}')
        creation = t.reshape(1,).repeat(x.shape[0],axis=0)
        merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
        u_pred = merged_model.forward_flow_pred_function(x, creation, train=False)
        return u_pred
    out = O.odeint(f, init_x, jnp.linspace(t_min,1.0,see_steps), rtol=1e-4, atol=1e-4) 
    # out = O.odeint(f, init_x, jnp.linspace(t_min,1.0,see_steps), rtol=1e-5, atol=1e-5) 
    return out
    # return out, None
  
# NOTE: problem with diffrax is that it is imcompatible with JAX 0.4.27
# import diffrax as D
# def solve_diffeq_by_diffrax(init_x,state,see_steps:int=10,t_min:float=0.0):
#     def f(t, x):
#     # def f(x, t):
#         # assert t.shape == (), ValueError(f't shape: {t.shape}')
#         creation = t.reshape(1,).repeat(x.shape[0],axis=0)
#         merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
#         u_pred = merged_model.forward_flow_pred_function(x, creation, train=False)
#         return u_pred
#     term = D.ODETerm(f)
#     solver = D.Dopri5()
#     controller = D.PIDController(rtol=1e-5, atol=1e-5)
#     out = D.diffeqsolve(term, solver, y0=init_x, t0=t_min,t1=1.0, stepsize_controller=controller) 
#     # out = O.odeint(f, init_x, jnp.linspace(t_min,1,see_steps), rtol=1e-5, atol=1e-5) 
#     return out

def sample_by_diffeq(state: NNXTrainState, model, rng, n_sample,t_min:float=0.0):
    别传进去 = rng
    只能用一次, 别传进去 = jax.random.split(别传进去)
    init_x = jax.random.normal(只能用一次, (n_sample, model.image_size, model.image_size, model.out_channels))    
    # samples, nfe = solve_diffeq_by_O(init_x, state, see_steps=2, t_min=t_min) # [2, N, 32, 32, 3] # [1, N]
    samples = solve_diffeq_by_O(init_x, state, see_steps=2, t_min=t_min)
    # we extract nfe from samples
    
    images = samples[1]
    nfes = samples[2].mean(axis=(1,2,3)).astype(jnp.int32)
    # print('nfes :', nfes)
    # print('average nfe:', nfes.mean())
    # print('samples.shape:', samples_.shape)
    # print('nfe.shape:', nfe.shape)
    return images, nfes.mean()

# move this out from model for JAX compilation
def generate(state: NNXTrainState, model, rng, n_sample, t_state=None, verbose=False):
  """
  Generate samples from the model

  Here we tend to not use nnx.Rngs
  state: maybe a train state
  ---
  return shape: (n_sample, 32, 32, 3)
  """
  if model.ode_solver == 'O':
    return sample_by_diffeq(state,model,rng,n_sample, t_min=model.eps)

  # prepare schedule
  num_steps = model.n_T
  # initialize noise
  x_shape = (n_sample, model.image_size, model.image_size, model.out_channels)
  rng_used, rng = jax.random.split(rng, 2)
  # sample from prior
  x_prior = jax.random.normal(rng_used, x_shape, dtype=model.dtype)

  if model.sampler in ['euler', 'heun', "adaptive"]:
    x_i = x_prior
    def step_fn(i, inputs):
      x_i, rng = inputs
      rng_this_step = jax.random.fold_in(rng, i)
      rng_z, 别传进去 = jax.random.split(rng_this_step, 2)
      merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
      x_i = merged_model.sample_one_step(x_i, rng_z, i, t_state=t_state, verbose=verbose)
      outputs = (x_i, rng)
      return outputs
    if not verbose:
      input = (x_i, rng)
      outputs = jax.lax.fori_loop(0, num_steps, step_fn, input)
      return outputs[0]
    else:
      all_x = []
      # denoised = []
      all_t = []
      for i in range(num_steps):
        D = step_fn(i, (x_i, rng))
        x_i, t = D[0]
        rng = D[1]
        all_t.append(t)
        all_x.append(x_i)
      images = jnp.stack(all_x, axis=0)
      all_t = jnp.stack(all_t, axis=0)
      return images, all_t
  
  elif model.sampler in ['edm', 'edm-sde']:
    t_steps = model.compute_t(jnp.arange(num_steps), num_steps)
    t_steps = jnp.concatenate([t_steps, jnp.zeros((1,), dtype=model.dtype)], axis=0)  # t_N = 0; no need to round_sigma
    x_i = x_prior * t_steps[0]
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
  
  elif model.sampler == "ban": # sqa experiment
    # here num_steps mean we solve for "0, i, 2i, ..., 1/2, 1/2+2i, 1/2+4i, ..., 1-2/i"
    assert num_steps % 4 == 0
    x_i = x_prior
    def step_fn_euler(i, inputs):
      x_i, rng = inputs
      rng_this_step = jax.random.fold_in(rng, i)
      rng_z, 别传进去 = jax.random.split(rng_this_step, 2)
      merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
      # x_i = merged_model.sample_one_step_euler(x_i, i)
      x_i = merged_model.sample_one_step_euler(x_i, i+num_steps//2)
      # x_i, denoised = merged_model.sample_one_step_euler(x_i, i) # for debug
      outputs = (x_i, rng)
      return outputs
      # return outputs, denoised # for debug
    def step_fn_heun(i, inputs):
      x_i, rng = inputs
      rng_this_step = jax.random.fold_in(rng, i)
      rng_z, 别传进去 = jax.random.split(rng_this_step, 2)
      merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
      # x_i = merged_model.sample_one_step_heun(x_i, 2*i+num_steps//2)
      x_i = merged_model.sample_one_step_heun(x_i, 2*i)
      # x_i, denoised = merged_model.sample_one_step_heun(x_i, 2*i+num_steps//2)
      outputs = (x_i, rng)
      return outputs
      # return outputs, denoised # for debug

    # # first euler then heun
    # outputs = jax.lax.fori_loop(0, num_steps//2, step_fn_euler, (x_i, rng))
    # outputs = jax.lax.fori_loop(0, num_steps//4, step_fn_heun, outputs)

    # first heun then euler
    outputs = jax.lax.fori_loop(0, num_steps//4, step_fn_heun, (x_i, rng))
    outputs = jax.lax.fori_loop(0, num_steps//2, step_fn_euler, outputs)
    images = outputs[0]
    return images

    # # for debug
    # all_x = []
    # denoised = []
    # for i in range(num_steps//2):
    #   D = step_fn_euler(i, (x_i, rng))
    #   x_i, rng = D[0]
    #   denoised.append(D[1])
    #   all_x.append(x_i)
    # for i in range(num_steps//4):
    #   D = step_fn_heun(i, (x_i, rng))
    #   x_i, rng = D[0]
    #   denoised.append(D[1])
    #   all_x.append(x_i)
    # images = jnp.stack(all_x, axis=0)
    # denoised = jnp.stack(denoised, axis=0)
    # return images, denoised

    
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
    P_mean = -1.2, # P_mean and P_std are for EDM use
    n_T = 18,  # inference steps
    net_type = 'ncsnpp',
    dropout = 0.0,
    dtype = jnp.float32,
    use_aug_label = False,
    average_loss = False,
    eps=1e-3,
    h_init=0.035,
    sampler='euler',
    ode_solver='jax',
    no_condition_t=False,
    rngs=None,
    embedding_type='fourier',
    # beta_schedule='linear',
    # beta_start=1e-4,
    # beta_end=0.02,
    # num_diffusion_timesteps=1000,
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
    self.no_condition_t = no_condition_t
    self.rngs = rngs
    # self.beta_schedule = beta_schedule
    # self.beta_start = beta_start
    # self.beta_end = beta_end
    # self.num_diffusion_timesteps = num_diffusion_timesteps

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
        embedding_type=embedding_type,
        use_aug_label=self.use_aug_label,
        aug_label_dim=9,
        rngs=self.rngs)
    else:
      raise ValueError(f'Unknown net type: {self.net_type}')

    # # declare two networks
    # self.net = net_fn(name='net')
    # self.net_ema = net_fn(name='net_ema')
    # self.num_timesteps = num_diffusion_timesteps
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

  # def compute_alpha(self, t):
  #   """
  #   DDIM util function
  #   """
  #   betas = get_beta_schedule(self.beta_schedule, beta_start=self.beta_start, beta_end=self.beta_end, num_diffusion_timesteps=self.num_diffusion_timesteps)
  #   alpha = jnp.cumprod(1 - betas, axis=0)
  #   alpha = jnp.concatenate([jnp.ones((1,)), alpha], axis=0)
  #   a = jnp.take(alpha, t + 1).reshape(-1, 1, 1, 1)
  #   return a
    
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

  def sample_one_step(self, x_i, rng, i, t_state=None, verbose=False):

    if self.sampler == 'euler':
      x_next = self.sample_one_step_euler(x_i, i) 
    elif self.sampler == 'heun':
      x_next = self.sample_one_step_heun(x_i, i) 
    elif self.sampler == 'adaptive':
      x_next = self.sample_one_step_adaptive(x_i, i, t_state=t_state, verbose=verbose)
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
    # t_next = (i + 2) / self.n_T # for ban
    t_next = t_next * (1 - self.eps) + self.eps

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
    # return x_next, x_cur + u_pred * (1 - t_cur)

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
    # return x_next, x_i + u_pred * (1 - t)

  def sample_one_step_adaptive(self, x_i, i, t_state, verbose=False):
    # this is for no t
    # i: loop from 0 to self.n_T - 1
    merged_model = nn.merge(t_state.graphdef, t_state.params, t_state.rng_states, t_state.batch_stats, t_state.useless_variable_state)
    t = merged_model.forward(x_i)
    t_shape = jnp.repeat(t, x_i.shape[0]) # to ensure the shape is correct

    u_pred = self.forward_flow_pred_function(x_i, t_shape, train=False)

    # move one step
    dt = jnp.minimum(jnp.ones_like(t) * 0.02, t) # adaptive step size, max=0.02
    dt = jnp.where(dt < 0.005, 0, dt) # for small t, we don't move
    x_next = x_i + batch_mul(u_pred, dt)

    if verbose:
      return x_next, t
    else:
      return x_next
  
  def sample_one_step_edm_ode(self, x_i, i, t_steps):
    """
    edm's second order ODE solver
    """

    x_cur = x_i
    t_cur = t_steps[i]
    t_next = t_steps[i + 1]

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

  # def sample_one_step_DDIM(self, x_i, rng, t, next_t):
  #   """
  #   rng here is useless, if we set eta = 0
  #   """
  #   # we only implement 'generalized' here
  #   # we only implement 'skip_type=uniform' here
  #   at = self.compute_alpha(t.astype(jnp.int32))
  #   at_next = self.compute_alpha(next_t.astype(jnp.int32))

  #   eps = self.forward_DDIM_pred_function(x_i, t, train=False)
  #   # x0_t = (x_i - eps * jnp.sqrt(1 - at)) / jnp.sqrt(at)
  #   x0_t = batch_mul(x_i - batch_mul(eps, jnp.sqrt(1 - at)), 1. / jnp.sqrt(at))  # when eta=0, no need to add noise
  #   # when eta=0, no need to add noise
  #   c2 = jnp.sqrt(1 - at_next)
  #   # x_next = jnp.sqrt(at_next) * x0_t + c2 * eps
  #   x_next = batch_mul(x0_t, jnp.sqrt(at_next)) + batch_mul(eps, c2)
  #   return x_next
  #   # x_next = x0_t = x_i
  #   # print(at, at_next) # debug
  #   # return x_next, x0_t # debug

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

    t_cond = jnp.zeros_like(t) if self.no_condition_t else jnp.log(t * 999)
    u_pred = self.net(z, t_cond, augment_label=augment_label, train=train)
    return u_pred

  # def forward_DDIM_pred_function(self, z, t, augment_label=None, train: bool = True):  # DDIM
  #   t_cond = jnp.zeros_like(t) if self.no_condition_t else t
  #   eps_pred = self.net(z, t_cond, augment_label=augment_label, train=train)
  #   return eps_pred
  
  def forward_edm_denoising_function(self, x, sigma, augment_label=None, train: bool = True):  # EDM
    """
    code from edm
    for FM API use
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

  def forward(self, imgs, labels, augment_label, noise_batch, t_batch, train: bool = True):
    """
    You should first sample the noise and t and input them
    """
    imgs = imgs.astype(self.dtype)
    gt = imgs
    x = imgs
    bz = imgs.shape[0]

    assert noise_batch.shape == x.shape
    assert t_batch.shape == (bz,)
    # t_batch = t_batch.reshape(bz, 1, 1, 1)

    # -----------------------------------------------------------------

    # sample from data
    x_data = x
    # sample from prior (noise)
    x_prior = noise_batch

    # sample t step
    t = t_batch
    t = t * (1 - self.eps) + self.eps
    # TODO: 这个太唐了，必须移到外面去

    # create v target
    v_target = x_data - x_prior
    # v_target = jnp.ones_like(x_data)  # dummy

    # create z (as the network input)
    z = batch_mul(t, x_data) + batch_mul(1 - t, x_prior)

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
