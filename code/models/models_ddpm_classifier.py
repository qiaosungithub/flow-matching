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
from models.models_ncsnpp_edm import NCSNppClassifier as NCSNppEDMClassifier
# from models.models_ncsnpp import NCSNpp
# import models.jcm.sde_lib as sde_lib
from models.jcm.sde_lib import batch_mul


ModuleDef = Any


class NNXTrainState(FlaxTrainState):
  batch_stats: Any
  rng_states: Any
  graphdef: Any
  useless_variable_state: Any
  # NOTE: is_training can't be a attr, since it can't be replicated


def get_t_process_fn(t_condition_method):
  if t_condition_method == 'log999':
    return lambda t: jnp.log(t * 999)
  elif t_condition_method == 'direct':
    return lambda t: t
  elif t_condition_method == 'not': # no t: 没有t
    return lambda t: jnp.zeros_like(t)
  else:
    raise NotImplementedError('Unknown t_condition_method: {m}'.format(m=t_condition_method))

def batch_t(t,b):
  return t.reshape(1,).repeat(b, axis=0)

class SimDDPM(nn.Module):
  """Simple DDPM Classifier"""

  def __init__(self,
    image_size,
    base_width,
    num_classes = -1,
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
    learn_var=False,
    class_conditional=False,
    classifier_model_depth=2,
    **kwargs
  ):
    self.image_size = image_size
    self.base_width = base_width
    self.num_classes = num_classes
    assert num_classes > 0, 'classifier!'
    assert learn_var == False, 'classifier!'
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
    self.learn_var = learn_var
    # self.no_condition_t = no_condition_t
    assert no_condition_t == False, 'This is deprecated'
    self.t_preprocess_fn = get_t_process_fn(t_condition_method)
    self.rngs = rngs
    self.sample_clip_denoised = sample_clip_denoised
    self.class_conditional = class_conditional

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
      net_fn = partial(NCSNppEDMClassifier,
        base_width=self.base_width,
        image_size=self.image_size,
        out_channels=self.out_channels,
        out_channel_multiple = (2 if self.learn_var else 1),
        dropout=self.dropout,
        class_conditional=self.class_conditional,
        num_classes=self.num_classes,
        num_res_blocks=classifier_model_depth,
        ### these settings are from repo
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_attn_pool=True,
        ### end
        rngs=self.rngs)
    else:
      raise ValueError(f'Unknown net type: {self.net_type}')

    # # declare two networks
    # self.net = net_fn(name='net')
    # self.net_ema = net_fn(name='net_ema')
    self.net = net_fn()
    
  def forward_prediction_function(self, z, t, augment_label=None,y=None, train: bool = True):  # EDM

    # t_cond = jnp.zeros_like(t) if self.no_condition_t else jnp.log(t * 999)
    t_cond = self.t_preprocess_fn(t).astype(self.dtype)
    u_pred = self.net(z, t_cond, train=train)
    return u_pred

  def get_visualization(self, list_imgs):
    vis = jnp.concatenate(list_imgs, axis=1)
    return vis

  def forward(self, imgs, labels, augment_label, noise_batch, t_batch,alpha_cumprod_batch, alpha_cumprod_prev_batch, posterior_log_variance_clipped_batch, beta_batch, train: bool = True):
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
    # assert (labels is None) or labels.shape == (bz,)
    assert (labels is not None) and labels.shape == (bz,)
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

    x_mixtue = batch_mul(sqrt_alphas_cumprod, x_data) + batch_mul(sqrt_one_minus_alphas_cumprod, x_prior)
    
    t = t_batch

    # create v target
    # v_target = x_prior
    display_only_labels = jax.nn.one_hot(labels * 3, 32) # [b, 32]
    v_target = display_only_labels.reshape(bz, 32, 1, 1).repeat(32,axis=2).repeat(3,axis=3)
    # v_target = jnp.ones_like(x_data)  # dummy


    # forward network
    u_pred = self.forward_prediction_function(x_mixtue, t)

    # loss
    # loss = (v_target - u_pred)**2
    # cross entropy loss
    one_hot_labels = jax.nn.one_hot(labels, self.num_classes)
    xentropy = optax.softmax_cross_entropy(u_pred, one_hot_labels)
    loss = jnp.mean(xentropy)

    loss_train = loss
    
    # vis & log
    best_pred = jnp.argmax(u_pred, axis=-1)
    sftmx_pred = jax.nn.softmax(u_pred, axis=-1)
    sftmx_pred = jnp.concatenate([sftmx_pred, jnp.zeros((bz, 20))], axis=-1).reshape(bz, 3, 10).transpose((0, 2, 1)).reshape(bz, 30)
    sftmx_pred = jnp.concatenate([sftmx_pred, jnp.zeros((bz, 2))], axis=-1)
    best_pred_img = jax.nn.one_hot(best_pred * 3, 32).reshape(bz, 32, 1, 1).repeat(32,axis=2).repeat(3,axis=3)
    sftmx_pred_img = sftmx_pred.reshape(bz, 32, 1, 1).repeat(32,axis=2).repeat(3,axis=3)

    dict_losses = {}
    dict_losses['loss'] = loss  # legacy
    dict_losses['loss_train'] = loss_train
    dict_losses['acc_train'] = jnp.mean(best_pred == labels)

    images = self.get_visualization(
      [gt,        # image (from dataset)
       x_mixtue,  # mixture of data and noise
       (v_target*2-1).transpose((0,2,1,3)).astype(jnp.float32),  # target of network (known)
       (best_pred_img*2-1).transpose((0,2,1,3)).astype(jnp.float32),    # argmax pred of network
       (sftmx_pred_img*2-1).transpose((0,2,1,3)).astype(jnp.float32),   # softmax pred of network
      ])

    return loss_train, dict_losses, images

  def __call__(self, imgs, labels, train: bool = False):
    # initialization only
    t = jnp.ones((imgs.shape[0],))
    augment_label = jnp.ones((imgs.shape[0], 9)) if self.use_aug_label else None  # fixed augment_dim # TODO: what is this?
    out = self.net(imgs, t, augment_label) # TODO: whether to add train=train
    # out = self.net(imgs, t, augment_label,y=jnp.ones((imgs.shape[0],))) # TODO: whether to add train=train
    out_ema = None   # no need to initialize it here
    return out, out_ema
