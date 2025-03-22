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

from typing import Any

import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState as FlaxTrainState

from functools import partial, reduce

# from models.models_unet import ContextUnet
from models.models_ncsnpp_edm import NCSNpp as NCSNppEDM
from models.jcm.sde_lib import batch_mul

from utils.logging_util import log_for_0

def compose(*funcs):
  return lambda x: reduce(lambda v, f: f(v), funcs, x)

class NNXTrainState(FlaxTrainState):
  batch_stats: Any
  rng_states: Any
  graphdef: Any
  useless_variable_state: Any
  # NOTE: is_training can't be a attr, since it can't be replicated

def edm_ema_scales_schedules(step, config, steps_per_epoch):
  ema_halflife_kimg = config.get("ema_kimg", 500)  # from edm
  ema_halflife_nimg = ema_halflife_kimg * 1000

  ema_rampup_ratio = config.get("ema_rampup_ratio", 0.05)
  ema_halflife_nimg = jnp.minimum(ema_halflife_nimg, step * config.batch_size * ema_rampup_ratio)

  ema_beta = 0.5 ** (config.batch_size / jnp.maximum(ema_halflife_nimg, 1e-8))
  scales = jnp.ones((1,), dtype=jnp.int32)
  return ema_beta, scales


# move this out from model for JAX compilation
def generate(state: NNXTrainState, model, rng, n_sample):
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
    raise NotImplementedError
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
  elif model.sampler in ['edm-euler']:
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
  elif model.sampler == 'DDIM':
    raise NotImplementedError
    skip = model.num_diffusion_timesteps // num_steps
    # skip = 1
    seq = range(0, model.num_timesteps, skip)
    T = len(seq)
    seq_next = [-1] + list(seq[:-1])
    seq_reversed = jnp.array(list(reversed(seq)))
    seq_next_reversed = jnp.array(list(reversed(seq_next)))
    eta = 0.0 # control the noise level added every step, 0 -> ODE sampler TODO: implement eta > 0.0
    n = x_prior.shape[0]
    x0_preds = []
    xs = [x_prior]

    x_i = x_prior

    def step_fn(i, inputs):
      x_i, rng = inputs

      _i = seq_reversed[i]
      _j = seq_next_reversed[i]

      t = jnp.ones(n) * _i
      next_t = jnp.ones(n) * _j

      rng_this_step = jax.random.fold_in(rng, i)
      rng_z, 别传进去 = jax.random.split(rng_this_step, 2)

      merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
      x_i = merged_model.sample_one_step_DDIM(x_i, rng_z, t, next_t)
      # x_i, denoised = merged_model.sample_one_step_DDIM(x_i, rng_z, t, next_t) # for debug

      outputs = (x_i, rng)
      return outputs
      # return outputs, denoised # for debug

    outputs = jax.lax.fori_loop(0, T, step_fn, (x_i, rng))
    images = outputs[0]
    return images
    # all_x = []
    # denoised = []
    # for i in range(T):
    #   D = step_fn(i, (x_i, rng))
    #   x_i, rng = D[0]
    #   denoised.append(D[1])
    #   all_x.append(x_i)
    # images = jnp.stack(all_x, axis=0)
    # denoised = jnp.stack(denoised, axis=0)
    # return images, denoised # for debug

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
    sampler='euler',
    ode_solver='jax',
    no_condition_t=False,
    rngs=None,
    double_temb=False,
    rho=7.0,
    t_cond_method: str = "edm", # options: ['edm', 'not']
    train_t_dropout: float = 0.0,
    sample_use_t: bool = True,
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
    self.sampler = sampler
    self.ode_solver = ode_solver
    self.no_condition_t = no_condition_t
    self.rngs = rngs
    self.double_temb = double_temb
    self.t_cond_method = t_cond_method
    self.train_t_dropout = train_t_dropout
    self.sample_use_t = sample_use_t

    assert no_condition_t == False, "this is deprecated"

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
        use_aug_label=self.use_aug_label,
        aug_label_dim=9,
        rngs=self.rngs,
        double_temb=double_temb,
      )
    else:
      raise ValueError(f'Unknown net type: {self.net_type}')

    # self.num_timesteps = num_diffusion_timesteps
    self.net = net_fn()

    if self.t_cond_method == "not": raise NotImplementedError
    log_for_0(f"using train_t_dropout: {self.train_t_dropout}")
    self.t_conder = (
      log_for_0("Use t-cond: edm (0.25logt)")  or (lambda t: 0.25*jnp.log(t))
    ) if self.t_cond_method == "edm" else (
      log_for_0("Use t-cond: not")  or (lambda t: t * 0.0)
    ) if self.t_cond_method == "not" else (
      log_for_0("Use t-cond: ???") or exec(f"raise ValueError('Unknown t_cond_method: {self.t_cond_method}')")
    )
    self.t_conder = compose(self.t_conder, lambda x: x.reshape(x.shape[0]))

    self.data_std = 0.5
    self.t_min = 0.002
    self.t_max = 80.0
    self.rho = rho


  def get_visualization(self, list_imgs):
    vis = jnp.concatenate(list_imgs, axis=1)
    return vis

  def compute_t(self, indices, scales):
    t = self.t_max ** (1 / self.rho) + indices / (scales - 1) * (
        self.t_min ** (1 / self.rho) - self.t_max ** (1 / self.rho)
    )
    t = t**self.rho
    return t

  def sample_t_conder(self, t):
    if self.sample_use_t:
      return self.t_conder(t)
    return jnp.zeros_like(t)

  def sample_one_step(self, x_i, rng, i):
    raise NotImplementedError
    if self.sampler == 'euler':
      x_next = self.sample_one_step_euler(x_i, i) 
    elif self.sampler == 'heun':
      x_next = self.sample_one_step_heun(x_i, i)
    else:
      raise NotImplementedError

    return x_next
  
  def sample_one_step_edm(self, x_i, rng, i, t_steps):

    if self.sampler == 'edm':
      x_next = self.sample_one_step_edm_ode(x_i, i, t_steps) 
      # x_next, denoised = self.sample_one_step_edm_ode(x_i, i, t_steps) # for debug
    elif self.sampler == 'edm-sde':
      raise NotImplementedError
      x_next = self.sample_one_step_edm_sde(x_i, rng, i, t_steps)
      # x_next, denoised = self.sample_one_step_edm_sde(x_i, rng, i, t_steps) # for debug
    elif self.sampler == 'edm-euler':
      raise NotImplementedError
      x_next = self.sample_one_step_edm_euler(x_i, i, t_steps)
    else:
      raise NotImplementedError

    return x_next
    # return x_next, denoised 

  def sample_one_step_heun(self, x_i, i):
    raise NotImplementedError
    x_cur = x_i

    t_cur = i / self.n_T  # t start from 0 (t = 0 is noise here)
    t_cur = t_cur * (1 - self.eps) + self.eps

    t_next = (i + 1) / self.n_T  # t start from 0 (t = 0 is noise here)
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

  def sample_one_step_euler(self, x_i, i):
    raise NotImplementedError
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

    t_hat = t_cur
    x_hat = x_cur  # x_hat is always x_cur when gamma=0

    t_hat = jnp.repeat(t_hat, x_hat.shape[0])
    t_next = jnp.repeat(t_next, x_hat.shape[0])
    
    # Euler step.
    t_cond_hat = self.sample_t_conder(t_hat)
    denoised = self.forward_edm_denoising_function(x_hat, t_hat, train=False, t_cond=t_cond_hat)
    d_cur = batch_mul(x_hat - denoised, 1. / t_hat)
    x_next = x_hat + batch_mul(d_cur, t_next - t_hat)

    # Apply 2nd order correction
    t_cond_next = self.sample_t_conder(t_next)
    denoised = self.forward_edm_denoising_function(x_next, t_next, train=False, t_cond=t_cond_next)
    d_prime = batch_mul(x_next - denoised, 1. / jnp.maximum(t_next, 1e-8)) # won't take effect if t_next is 0 (last step)
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

  def sample_one_step_edm_euler(self, x_i, i, t_steps):
    """
    Euler with EDM t schedule
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

    # return x_next, denoised # for debug
    return x_next
  
  def forward_edm_denoising_function(self, x, sigma, t_cond, augment_label=None, train=True):  # EDM
    """
    code from edm
    ---
    input: x (noisy image, =x+sigma*noise), sigma (condition)
    We hope this function operates D(x+sigma*noise) = x
    our network has F((1-t)x + t*noise) = x - noise
    """

    # # use FM network to denoise
    # c_in = 1 / (sigma + 1)
    # in_x = batch_mul(x, c_in)
    # c_out = sigma / (sigma + 1)

    # F_x = self.forward_flow_pred_function(in_x, c_in, augment_label=augment_label, train=train)

    # D_x = in_x + batch_mul(F_x, c_out)
    # return D_x

    # edm network
    c_skip = self.data_std ** 2 / (sigma ** 2 + self.data_std ** 2)
    c_out = sigma * self.data_std / jnp.sqrt(sigma ** 2 + self.data_std ** 2)
    # c_out = jnp.ones_like(sigma) # Kaiming shenyi

    c_in = 1 / jnp.sqrt(sigma ** 2 + self.data_std ** 2)
    # c_in = 1 / jnp.sqrt(sigma ** 2 + 1) # Kaiming shenyi
    # c_noise = jnp.zeros_like(sigma) if self.no_condition_t else 0.25 * jnp.log(sigma)

    # forward network
    in_x = batch_mul(x, c_in)
    # c_noise = c_noise.reshape(c_noise.shape[0])

    F_x = self.net(in_x, t_cond, augment_label=augment_label, train=train)

    D_x = batch_mul(x, c_skip) + batch_mul(F_x, c_out)
    return D_x

  def forward(self, imgs, labels, augment_label, noise_batch, t_batch, t_mask):
    """
    edm version
    ---
    You should first sample the noise and t and input them
    ---
    t_batch: here is normal (bs,), we will process it into sigma batch. NOTE: this is normal, not uniform
    """
    imgs = imgs.astype(self.dtype)
    gt = imgs
    x = imgs
    bz = imgs.shape[0]

    assert noise_batch.shape == x.shape
    assert t_batch.shape == (bz,)
    # t_batch = t_batch.reshape(bz, 1, 1, 1)

    # -----------------------------------------------------------------
    # sample t step
    sigma = jnp.exp(t_batch * self.P_std + self.P_mean)
    weight = (sigma ** 2 + self.data_std ** 2) / (sigma * self.data_std) ** 2

    xn = x + batch_mul(noise_batch, sigma)
    t_cond = self.t_conder(sigma)
    # t_cond = t_cond * t_mask

    # make sure the dropout is the same for t forward and w/o t forward
    rng_dropout_this_step = self.rngs.dropout()
    nn.reseed(self, dropout=rng_dropout_this_step)
    D_xn_t = self.forward_edm_denoising_function(xn, sigma, augment_label=augment_label, t_cond=t_cond)
    nn.reseed(self, dropout=rng_dropout_this_step)
    D_xn_wot = self.forward_edm_denoising_function(xn, sigma, augment_label=augment_label, t_cond=jnp.zeros_like(t_cond))

    # loss
    mse_loss = (D_xn_t - gt)**2
    con_loss = (jax.lax.stop_gradient(D_xn_t) - D_xn_wot)**2
    loss = mse_loss * t_mask + con_loss * (1 - t_mask)
    loss = batch_mul(loss, weight)

    if self.average_loss:
      raise ValueError("we recommend to use sum loss")
      loss = jnp.mean(loss, axis=(1, 2, 3))  # mean over pixels
    else:
      loss = jnp.sum(loss, axis=(1, 2, 3))  # sum over pixels
    loss = loss.mean()  # mean over batch

    loss_train = loss

    dict_losses = {}
    dict_losses['mse_loss'] = mse_loss
    dict_losses['con_loss'] = con_loss
    dict_losses['loss_train'] = loss_train

    # prepare some visualization
    # if we can pred u, then we can reconstruct x_data from x_prior
    images = self.get_visualization([gt, xn, D_xn_t, D_xn_wot])

    return loss_train, dict_losses, images

  def __call__(self, imgs, labels, train: bool = False):
    # initialization only
    t = jnp.ones((imgs.shape[0],))
    augment_label = jnp.ones((imgs.shape[0], 9)) if self.use_aug_label else None  # fixed augment_dim # TODO: what is this?
    out = self.net(imgs, t, augment_label)
    out_ema = None   # no need to initialize it here
    return out, out_ema
