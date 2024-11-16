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

from flax import linen as nn
import jax
import jax.numpy as jnp

import numpy as np
import optax

from functools import partial

# from models.models_unet import ContextUnet
from models.models_zyflow import ZYFlow
# import models.jcm.sde_lib as sde_lib
from models.jcm.sde_lib import batch_mul


ModuleDef = Any


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
  ema_halflife_kimg = 500  # from edm
  ema_halflife_nimg = ema_halflife_kimg * 1000

  ema_rampup_ratio = 0.05
  ema_halflife_nimg = jnp.minimum(ema_halflife_nimg, step * config.batch_size * ema_rampup_ratio)

  ema_beta = 0.5 ** (config.batch_size / jnp.maximum(ema_halflife_nimg, 1e-8))
  scales = jnp.ones((1,), dtype=jnp.int32)
  return ema_beta, scales


# move this out from model for JAX compilation
def generate(params, model, rng, n_sample):
  """
  Generate samples from the model
  """

  # prepare schedule
  num_steps = model.n_T
  t_steps = model.apply(
      {},
      num_steps=num_steps,
      method=model.sampling_schedule(),
  )

  # initialize noise
  x_shape = (n_sample, model.image_size, model.image_size, model.out_channels)
  rng_used, rng = jax.random.split(rng, 2)
  latents = jax.random.normal(rng_used, x_shape, dtype=model.dtype)  # x_T ~ N(0, 1), sample initial noise
  
  x_i = latents

  def step_fn(i, inputs):
    x_i, rng = inputs
    rng_this_step = jax.random.fold_in(rng, i)
    rng_z, rng_dropout = jax.random.split(rng_this_step, 2)
    x_i, _ = model.apply(
        params,  # which is {'params': state.params, 'batch_stats': state.batch_stats},
        x_i, rng_z, i, t_steps,
        # rngs={'dropout': rng_dropout},  # we don't do dropout in eval
        rngs={},
        method=model.sample_one_step,
        mutable=['batch_stats'],
    )
    outputs = (x_i, rng)
    return outputs

  outputs = jax.lax.fori_loop(0, num_steps, step_fn, (x_i, rng))
  images = outputs[0]
  return images


class SimFlow(nn.Module):
  """Simple Flow."""
  image_size: int
  base_width: int
  num_classes: int = 10
  out_channels: int = 1

  net_type: str = 'flow'
  embedding_type: str = 'fourier'
  fir: bool = True
  resblock_type: str = 'biggan'
  progressive_input: str = 'residual'

  dropout: float = 0.0
  dtype: Any = jnp.float32
  use_aug_label: bool = False

  # Inference setups
  n_T: int = 18
  solver: str = 'heun'
  schedule: str = 'uniform'

  # Noise Distribution
  noise_dist: str = 'logit_normal'
  P_mean: float = 0.0
  P_std: float = 1.0

  # Adaptive weighting
  loss_eps: float = 0.
  loss_adp: float = 0.

  def setup(self):

    if self.net_type == 'flow':
      net_fn = partial(ZYFlow,
        base_width=self.base_width,
        image_size=self.image_size,
        embedding_type=self.embedding_type,
        resblock_type=self.resblock_type,
        progressive_input=self.progressive_input,
        fir=self.fir,
        dropout=self.dropout)
    else:
      raise ValueError(f'Unknown net type: {self.net_type}')

    # declare two networks
    self.net = net_fn(name='net')
    self.net_ema = net_fn(name='net_ema')
    
  def compute_losses(self, pred, gt):
    assert pred.shape == gt.shape

    # simple l2 loss
    loss = (pred - gt) ** 2
    loss_rec = jnp.mean(loss)
    
    loss_train = loss_rec

    dict_losses = {
      'loss_rec': loss_rec,
      'loss_train': loss_train
    }
    return loss_train, dict_losses

  def solver_step(self):
    if self.solver == 'euler':
        return self._euler_solver
    elif self.solver == 'heun':
        return self._heun_solver
    else:
        raise ValueError(f"Unknown solver: {solver}")

  def _euler_solver(self, x_t: jnp.ndarray, t: float, r: float, i: int):
    v_t = self.forward_flow_function(x_t, t, train=False)
    return x_t + batch_mul(t - r, v_t)

  def _heun_solver(self, x_t: jnp.ndarray, t: float, r: float, i: int):
    v_t = self.forward_flow_function(x_t, t, train=False)
    x_euler = x_t + batch_mul(t - r, v_t)
    
    v_r = self.forward_flow_function(x_euler, r, train=False)
    x_heun = x_t + batch_mul((t - r) * 0.5, (v_t + v_r))

    return jnp.where(i < self.n_T - 1, x_heun, x_euler)

  def sample_one_step(self, x_t, rng, i, t_steps):
    t = t_steps[i]
    r = t_steps[i + 1]

    # TODO{kaiming}: revisit S_churn
    t = jnp.repeat(t, x_t.shape[0])
    r = jnp.repeat(r, x_t.shape[0])
    
    # TODO: ema net?
    # Solver step.
    x_next = self.solver_step()(x_t, t, r, i)
    return x_next

  def sampling_schedule(self):
    if self.schedule == 'uniform':
      return self._uniform_schedule
    elif self.schedule == 'quadratic':
      return self._quadratic_schedule
    elif self.schedule == 'inv_quadratic':
      return self._inv_quadratic_schedule
    elif self.schedule == 'cubic':
      return self._cubic_schedule
    elif self.schedule == 'cosine':
      return self._cosine_schedule
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

  def _uniform_schedule(self, num_steps):
    return jnp.linspace(1.0, 0.0, num_steps + 1)

  def _quadratic_schedule(self, num_steps):
    t = jnp.linspace(1.0, 0.0, num_steps + 1)
    return t**2

  def _cubic_schedule(self, num_steps):
    t = jnp.linspace(1.0, 0.0, num_steps + 1)
    return t**3

  def _inv_quadratic_schedule(self, num_steps):
    t = jnp.linspace(0.0, 1.0, num_steps + 1)
    return 1 - t**2

  def _cosine_schedule(self, num_steps):
    t = jnp.linspace(0.0, 1.0, num_steps + 1)
    return jnp.cos(0.5 * jnp.pi * t)

  def noise_distribution(self):
    if self.noise_dist == 'logit_normal':
        return self._logit_normal_dist
    elif self.noise_dist == 'uniform':
        return self._uniform_dist
    else:
        raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

  def _logit_normal_dist(self, bz):
    rnd_normal = jax.random.normal(self.make_rng('gen'), [bz, 1, 1, 1], dtype=self.dtype)
    return nn.sigmoid(
      rnd_normal * self.P_std + self.P_mean
      )
  
  def _uniform_dist(self, bz):
    return jax.random.uniform(self.make_rng('gen'), [bz, 1, 1, 1], dtype=self.dtype)

  def forward_flow_function(self, x, t, augment_label=None, train: bool = True):
    return self.net(x, t.reshape(t.shape[0]), augment_label=augment_label, train=train)

  def forward(self, imgs, labels, augment_label, train: bool = True):
    imgs = imgs.astype(self.dtype)
    bz = imgs.shape[0]
    x_0 = imgs

    # -----------------------------------------------------------------
    t = self.noise_distribution()(bz)

    # TODO(zygeng): Add weighting functions here.

    noise = jax.random.normal(self.make_rng('gen'), x_0.shape, dtype=self.dtype)
    x_t = (1 - t) * x_0 + t * noise
    v_gt = x_0 - noise

    v_t = self.forward_flow_function(x_t, t, augment_label)

    loss = (v_t - v_gt) ** 2
    loss = jnp.sum(loss, axis=(1, 2, 3))  # sum over pixels

    # Adaptive Weighting (static branching)
    if self.loss_adp > 0:
      loss = loss / (jax.lax.stop_gradient(loss) + self.loss_eps) ** self.loss_adp
    
    loss = loss.mean()  # mean over batch
    
    loss_train = loss

    dict_losses = {}
    dict_losses['loss'] = loss  # legacy
    dict_losses['loss_train'] = loss_train

    x0_pred = x_t + batch_mul(t, v_t)
    images = self.get_visualization([x_0, x_t, x0_pred])

    return loss_train, dict_losses, images

  def get_visualization(self, list_imgs):
    vis = jnp.concatenate(list_imgs, axis=1)
    return vis

  def __call__(self, imgs, labels, train: bool = False):
    # initialization only
    t = jnp.ones((imgs.shape[0],))
    augment_label = jnp.ones((imgs.shape[0], 9)) if self.use_aug_label else None  # fixed augment_dim
    out = self.net(imgs, t, augment_label)
    out_ema = None   # no need to initialize it here
    return out, out_ema
