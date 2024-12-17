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
from models.models_ncsnpp_edm import NCSNpp as NCSNppEDM, NCSNppClassifier as NCSNppEDMClassifier, SQA_TNet_Ver1 as SQA_TNet_Ver1
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
  ema_halflife_kimg = config.get('ema_halflife_king',50000)  # log(0.5) / log(0.999999) * 128 / 1000 = 88722 kimg, from flow
  ema_halflife_nimg = ema_halflife_kimg * 1000

  ema_rampup_ratio = 0.05
  ema_halflife_nimg = jnp.minimum(ema_halflife_nimg, step * config.batch_size * ema_rampup_ratio)

  ema_beta = 0.5 ** (config.batch_size / jnp.maximum(ema_halflife_nimg, 1e-8))
  scales = jnp.ones((1,), dtype=jnp.int32)
  return ema_beta, scales

def general_ema_scales_schedules(step, config, steps_per_epoch):
  sche_type = config.get('ema_schedule', 'edm')
  if sche_type in ['edm', 'EDM']:
    return edm_ema_scales_schedules(step, config, steps_per_epoch)
  else:
    raise NotImplementedError(f'Unknown ema_schedule: {sche_type}')

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Diffusion utility fn, copied from some repo not remembered
    
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
  
def space_timesteps(num_timesteps, section_counts):
    """
    Diffusion utility fn, copied from some repo not remembered
    
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return list(sorted(set(all_steps))) # this is something like [0, 4, 8, ..., 999]

def everything_from_beta(betas):
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    alphas_cumprod_prev = jnp.append(1.0, alphas_cumprod[:-1])
    alphas_cumprod_next = jnp.append(alphas_cumprod[1:], 0.0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = jnp.log(jnp.append(posterior_variance[1], posterior_variance[1:]))
    return {
      'betas': betas, 'alphas': alphas, 'alphas_cumprod': alphas_cumprod, 'alphas_cumprod_prev': alphas_cumprod_prev, 'alphas_cumprod_next': alphas_cumprod_next,
      'posterior_variance': posterior_variance, 'posterior_log_variance_clipped': posterior_log_variance_clipped,
    }    
  
def diffusion_sampling_schedule(diffusion_schedule, diffusion_nT, sample_nT):
    diffusion_steps_total = diffusion_nT
    sample_steps_total = sample_nT
    
    o = everything_from_beta(diffusion_beta_schedule(diffusion_schedule, diffusion_steps_total))
    sample_ts = space_timesteps(diffusion_steps_total,str(sample_steps_total))[::-1]
    
    new_betas = []
    new_alpha_cumprods = []
    last_alpha_cumprod = 1.0
    for i, alpha_cumprod in enumerate(o['alphas_cumprod']):
      if i in sample_ts:
          new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
          last_alpha_cumprod = alpha_cumprod
          new_alpha_cumprods.append(alpha_cumprod)
          # self.timestep_map.append(i)
    new_betas = jnp.array(new_betas)
    new_alpha_cumprods = jnp.array(new_alpha_cumprods)
    new_alphas_cumprod_prev = jnp.append(1.0, new_alpha_cumprods[:-1])
    new_alphas_cumprod_next = jnp.append(new_alpha_cumprods[1:], 0.0)
          
    # 更多的大便
    posterior_variance = new_betas * (1.0 - new_alphas_cumprod_prev) / (1.0 - new_alpha_cumprods)
    posterior_log_variance_clipped = jnp.log(jnp.append(posterior_variance[1], posterior_variance[1:]))
    model_variance = jnp.append(posterior_variance[1],new_betas[1:])
    log_model_variance = jnp.log(model_variance)
    sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / new_alpha_cumprods)
    sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / new_alpha_cumprods - 1.0)
    posterior_mean_coef1 = (
        new_betas * jnp.sqrt(new_alphas_cumprod_prev) / (1.0 - new_alpha_cumprods)
    )
    posterior_mean_coef2 = (
        (1.0 - new_alphas_cumprod_prev)
        * jnp.sqrt(1-new_betas)
        / (1.0 - new_alpha_cumprods)
    )
    
    # finally convert to jax, to avoid stange
    sample_ts = jnp.array(sample_ts)
    o.update({
        'sample_ts': sample_ts,
        'sample_posterior_variance': posterior_variance[::-1],
        'sample_posterior_log_variance_clipped': posterior_log_variance_clipped[::-1],
        'sample_model_variance': model_variance[::-1],
        'sample_log_model_variance': log_model_variance[::-1],
        'sample_sqrt_recip_alphas_cumprod': sqrt_recip_alphas_cumprod[::-1],
        'sample_sqrt_recipm1_alphas_cumprod': sqrt_recipm1_alphas_cumprod[::-1],
        'sample_posterior_mean_coef1': posterior_mean_coef1[::-1],
        'sample_posterior_mean_coef2': posterior_mean_coef2[::-1],
        'sample_beta': new_betas[::-1],
        'sample_alpha_cumprods': new_alpha_cumprods[::-1],
        'sample_alphas_cumprod_prev': new_alphas_cumprod_prev[::-1],
    }) # note we flip t
    return o

# utility functions
def index_dictionary(d, idx):
    return {k: v[idx] for k, v in d.items()}
  
def batch_t(t,b):
    return t.reshape(1,).repeat(b, axis=0)
  
def diffusion_beta_schedule(diffusion_schedule, diffusion_nT):
    if diffusion_schedule == 'cosine':
      return betas_for_alpha_bar(
              diffusion_nT,
              lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
          )
    elif diffusion_schedule == 'linear':
      return jnp.linspace(
            1e-4, 0.02, diffusion_nT, dtype=np.float32
        )
    else:
      raise NotImplementedError(f'Unknown diffusion schedule: {diffusion_schedule}')
    

# from jax.experimental import ode as O
# import models.ode_pkg_repo as O
import models.ode_pkg as O
def solve_diffeq_by_O(init_x,state,see_steps:int=10,t_min:float=0.0):
    def f(x, t):
        # assert t.shape == (), ValueError(f't shape: {t.shape}')
        creation = t.reshape(1,).repeat(x.shape[0],axis=0)
        merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
        u_pred = merged_model.forward_prediction_function(x, creation, train=False)
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
#         u_pred = merged_model.forward_prediction_function(x, creation, train=False)
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
def generate(state: NNXTrainState, model, rng, n_sample, config, label_type='random',classifier=None,classifier_state=None,classifier_scale=0.0):
  """
  Generate samples from the model

  Here we tend to not use nnx.Rngs
  state: maybe a train state
  ---
  return shape: (n_sample, 32, 32, 3)
  """
  assert (classifier is None) == (classifier_state is None), 'classifier and classifier_state should be None or not None at the same time'
  if model.ode_solver == 'O':
    return sample_by_diffeq(state,model,rng,n_sample, t_min=model.eps)

  # prepare schedule
  num_steps = model.n_T

  # initialize noise
  x_shape = (n_sample, model.image_size, model.image_size, model.out_channels)
  别传进去 = rng
  只能用一次, 别传进去 = jax.random.split(别传进去, 2)
  # sample from prior
  x_prior = jax.random.normal(只能用一次, x_shape, dtype=model.dtype)
  
  只能用一次啊, 别传进去 = jax.random.split(别传进去, 2)
  # generate labels
  if label_type == 'random':
    y = jax.random.randint(只能用一次啊, (n_sample,), 0, model.num_classes)
  elif label_type == 'order':
    y = jnp.arange(n_sample) % model.num_classes
    # y = jnp.ones((n_sample,), dtype=jnp.int32) * 9
  elif label_type == 'none': # no condition
    y = None
  else:
    raise NotImplementedError(f'Unknown label type: {label_type}')

  只能用一次诶, 别传进去 = jax.random.split(别传进去, 2)
  rng = 只能用一次诶

  if model.sampler in ['euler', 'heun']:
    assert y is None, NotImplementedError()
    assert classifier is None, NotImplementedError()
      
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
    return images,  {
      'euler': num_steps,
      'heun': 2 * num_steps - 1,
    }[model.sampler] # heun has two steps per iteration
  
  elif model.sampler in ['edm', 'edm-sde']:
    assert y is None, NotImplementedError()
    assert classifier is None, NotImplementedError()
    t_steps = model.compute_edm_t(jnp.arange(num_steps), num_steps)
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
    return images, {
      'edm': 2 * num_steps - 1,
      'edm-sde': 2 * num_steps - 1,
    }[model.sampler]
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
  elif model.sampler in ['ddpm', 'DDPM']:
    x_i = x_prior
    o = model.sampling_diffusion_schedule()
    t_steps = o['sample_ts']
    sqrt_recip_alphas_cumprod_steps = o['sample_sqrt_recip_alphas_cumprod']
    sqrt_recipm1_alphas_cumprod_steps = o['sample_sqrt_recipm1_alphas_cumprod']
    posterior_mean_coef1_steps = o['sample_posterior_mean_coef1']
    posterior_mean_coef2_steps = o['sample_posterior_mean_coef2']
    log_model_variance_steps = o['sample_log_model_variance']
    posterior_log_variance_clipped_steps = o['sample_posterior_log_variance_clipped']
    beta_steps = o['sample_beta']
    
    def classifier_fn(x, y, t):
      b = x.shape[0]
      # x.shape: [b, 32, 32, 3]
      # y.shape: [b]
      # t.shape: [b]
      assert x.shape[1:] == (32, 32, 3), 'Get x.shape {s}'.format(s=x.shape)
      assert y.shape == (b,), 'Get y.shape {s}'.format(s=y.shape)
      assert t.shape == (b,), 'Get t.shape {s}'.format(s=t.shape)
      
      merged_classifier = nn.merge(classifier_state.graphdef, classifier_state.params, classifier_state.rng_states, classifier_state.batch_stats, classifier_state.useless_variable_state)
      logits = merged_classifier.forward_prediction_function(x,t)
      logits = jax.nn.log_softmax(logits, axis=-1)
      selected_logits = logits[jnp.arange(b), y]
      return selected_logits.sum()

    def step_fn(i, inputs):
      x_i, rng = inputs
      rng_this_step = jax.random.fold_in(rng, i)
      rng_z, 别传进去 = jax.random.split(rng_this_step, 2)

      merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
      x_i = merged_model.sample_one_step_ddpm(x_i, rng_z, i, y=y, t_steps=t_steps, sqrt_recip_alphas_cumprod_steps=sqrt_recip_alphas_cumprod_steps, sqrt_recipm1_alphas_cumprod_steps=sqrt_recipm1_alphas_cumprod_steps,posterior_mean_coef1_steps=posterior_mean_coef1_steps,posterior_mean_coef2_steps=posterior_mean_coef2_steps,log_model_variance_steps=log_model_variance_steps,posterior_log_variance_clipped_steps=posterior_log_variance_clipped_steps,beta_steps=beta_steps,classifier_grad_fn=jax.grad(classifier_fn) if classifier is not None else None,classifier_scale=classifier_scale)
      outputs = (x_i, rng)
      return outputs

    # outputs = jax.lax.fori_loop(0, 1, step_fn, (x_i, rng))
    outputs = jax.lax.fori_loop(0, num_steps, step_fn, (x_i, rng))
    # outputs = jax.lax.fori_loop(0, num_steps-2, step_fn, (x_i, rng))
    images = outputs[0]
    
    # for debug
    # all_x = []
    # denoised = []
    # for i in range(num_steps):
    #   D = step_fn(i, (x_i, rng))
    #   (x_i, denoise), rng = D
    #   denoised.append(denoise)
    #   all_x.append(x_i)
    #   jax.debug.print('step {i} done', i=i)
    # images = jnp.stack(all_x, axis=0)
    # denoised = jnp.stack(denoised, axis=0)
    # return images, denoised
  
    return images, num_steps
  elif model.sampler in ['ddim', 'DDIM']:
    assert y is None, NotImplementedError()
    assert classifier is None, NotImplementedError()
    x_i = x_prior
    o = model.sampling_diffusion_schedule()
    t_steps = o['sample_ts']
    alpha_cumprod_steps = o['sample_alpha_cumprods']
    alpha_cumprod_prev_steps = o['sample_alphas_cumprod_prev']

    def step_fn(i, inputs):
      x_i, rng = inputs
      rng_this_step = jax.random.fold_in(rng, i)
      rng_z, 别传进去 = jax.random.split(rng_this_step, 2)

      merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
      x_i = merged_model.sample_one_step_DDIM(x_i, rng_z, i, y=y, t_steps=t_steps, alpha_cumprod_steps=alpha_cumprod_steps, alpha_cumprod_prev_steps=alpha_cumprod_prev_steps)
      outputs = (x_i, rng)
      return outputs
      # return outputs, denoised # for debug

    outputs = jax.lax.fori_loop(0, num_steps, step_fn, (x_i, rng))
    images = outputs[0]
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
    
    return images, num_steps
  else:
    raise NotImplementedError(f'Unknown sampler: {model.sampler}')

# loss utilities
def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = jnp.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = jnp.log(jnp.clip(cdf_plus,min=1e-12,max=None))
    log_one_minus_cdf_min = jnp.log(jnp.clip(1.0 - cdf_min,min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = jnp.where(
        x < -0.999,
        log_cdf_plus,
        jnp.where(x > 0.999, log_one_minus_cdf_min, jnp.log(jnp.clip(cdf_delta,min=1e-12))),
    )
    # assert log_probs.shape == x.shape
    return log_probs

def get_t_process_fn(t_condition_method):
  if t_condition_method == 'log999':
    return lambda t: jnp.log(t * 999)
  elif t_condition_method == 'direct':
    return lambda t: t
  elif t_condition_method == 'not': # no t: 没有t
    return lambda t: jnp.zeros_like(t)
  elif t_condition_method == 'dot25log':
    return lambda t: 0.25 * jnp.log(t)
  else:
    raise NotImplementedError('Unknown t_condition_method: {m}'.format(m=t_condition_method))

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
    # h_init=0.035,
    sampler='euler',
    sample_clip_denoised=True,
    ode_solver='jax',
    # no_condition_t=False,
    t_condition_method = 'log999',
    rngs=None,
    embedding_type='positional',
    # task
    task='FM',
    
    # diffusion
    diffusion_schedule='cosine',
    diffusion_nT=1000,
    
    # adm
    learn_var=False,
    class_conditional=False,
    
    # classifier
    classifier_model_depth=2,
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
    # self.h_init = h_init
    self.sampler = sampler
    self.ode_solver = ode_solver
    self.learn_var = learn_var
    # self.no_condition_t = no_condition_t
    self.t_preprocess_fn = get_t_process_fn(t_condition_method)
    self.task = task
    self.diffusion_schedule = diffusion_schedule
    self.diffusion_nT = diffusion_nT
    self.rngs = rngs
    logging.info(f'Model initialization Get additional kwargs: {kwargs}')
    # self.beta_schedule = beta_schedule
    # self.beta_start = beta_start
    # self.beta_end = beta_end
    # self.num_diffusion_timesteps = num_diffusion_timesteps
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
      net_fn = partial(NCSNppEDM,
        base_width=self.base_width,
        image_size=self.image_size,
        out_channels=self.out_channels,
        out_channel_multiple = (2 if self.learn_var else 1),
        dropout=self.dropout,
        embedding_type=embedding_type,
        use_aug_label=self.use_aug_label,
        aug_label_dim=9,
        class_conditional=self.class_conditional,
        num_classes=self.num_classes,
        rngs=self.rngs)
    elif self.net_type == 'ncsnppedm_classifier':
      net_fn = partial(NCSNppEDMClassifier,
        base_width=self.base_width,
        image_size=self.image_size,
        out_channels=self.out_channels,
        out_channel_multiple = (2 if self.learn_var else 1),
        dropout=self.dropout,
        # embedding_type=embedding_type, # this is uselsee
        use_aug_label=self.use_aug_label,
        aug_label_dim=9,
        class_conditional=self.class_conditional,
        num_classes=self.num_classes,
        num_res_blocks=classifier_model_depth,
        ### these settings are from repo
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_attn_pool=True,
        ### end
        rngs=self.rngs)
    elif self.net_type == 'ncsnppedm_t_predictor':
      assert not self.class_conditional, 'class conditional t predictors are not supported yet'
      net_fn = partial(NCSNppEDMClassifier,
        base_width=self.base_width,
        image_size=self.image_size,
        out_channels=self.out_channels,
        out_channel_multiple = (2 if self.learn_var else 1),
        dropout=self.dropout,
        # embedding_type=embedding_type, # this is uselsee
        use_aug_label=self.use_aug_label,
        aug_label_dim=9,
        class_conditional=self.class_conditional,
        num_classes=1,
        num_res_blocks=classifier_model_depth,
        ### these settings are from repo
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_attn_pool=True,
        ### end
        rngs=self.rngs)
    elif self.net_type == 'sqa_t_predictor':
      assert not self.class_conditional, 'class conditional t predictors are not supported yet'
      net_fn = partial(SQA_TNet_Ver1,
        base_width=self.base_width,
        # use_sigmoid = True,
        use_sigmoid = False,
        rngs=self.rngs)
    else:
      raise ValueError(f'Unknown net type: {self.net_type}')

    # # declare two networks
    # self.net = net_fn(name='net')
    # self.net_ema = net_fn(name='net_ema')
    # self.num_timesteps = num_diffusion_timesteps
    self.net = net_fn()
    
    # edm specific settings
    if self.task in ['edm', 'EDM']:
      self.t_min = 0.002
      self.t_max = 80.0
      self.rho = 7.0
      self.data_std = 0.5    
      
    if 't_predictor' in self.task:
      assert t_condition_method == 'not', 't predictors cannot condition on t'

  def get_visualization(self, list_imgs):
    vis = jnp.concatenate(list_imgs, axis=1)
    return vis

  def compute_edm_t(self, indices, scales):
    t = self.t_max ** (1 / self.rho) + indices / (scales - 1) * (
        self.t_min ** (1 / self.rho) - self.t_max ** (1 / self.rho)
    )
    t = t**self.rho
    return t

  def training_diffusion_schedule(self):
    """
    We trade-off speed (should be negligible) with the simplicity of implementation: this function is computed per forward call
    """
    return everything_from_beta(diffusion_beta_schedule(self.diffusion_schedule, self.diffusion_nT))
  
  def sampling_diffusion_schedule(self):
    return diffusion_sampling_schedule(self.diffusion_schedule, self.diffusion_nT, self.n_T)
    
  def compute_losses(self, pred, gt):
    raise NotImplementedError
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
    else:
      raise NotImplementedError(f'Unknown sampler: {self.sampler}')

    return x_next
  
  def sample_one_step_edm(self, x_i, rng, i, t_steps):

    if self.sampler == 'edm':
      x_next = self.sample_one_step_edm_ode(x_i, i, t_steps) 
      # x_next, denoised = self.sample_one_step_edm_ode(x_i, i, t_steps) # for debug
    elif self.sampler == 'edm-sde':
      x_next = self.sample_one_step_edm_sde(x_i, rng, i, t_steps)
      # x_next, denoised = self.sample_one_step_edm_sde(x_i, rng, i, t_steps) # for debug
    else:
      raise NotImplementedError(f'Unknown sampler: {self.sampler}')

    return x_next
    # return x_next, denoised 

  def sample_one_step_heun(self, x_i, i):

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
    u_pred = self.forward_prediction_function(x_i, t_hat, train=False)
    d_cur = u_pred
    x_next = x_hat + batch_mul(u_pred, t_next - t_hat)

    # Apply 2nd order correction
    u_pred = self.forward_prediction_function(x_next, t_next, train=False)
    d_prime = u_pred
    x_next_ = x_hat + batch_mul(0.5 * d_cur + 0.5 * d_prime, t_next - t_hat)

    x_next = jnp.where(i < self.n_T - 1, x_next_, x_next)

    return x_next

  def sample_one_step_euler(self, x_i, i):
    # i: loop from 0 to self.n_T - 1
    t = i / self.n_T  # t start from 0 (t = 0 is noise here)
    t = t * (1 - self.eps) + self.eps
    t = jnp.repeat(t, x_i.shape[0])

    u_pred = self.forward_prediction_function(x_i, t, train=False)

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

  def sample_one_step_ddpm(self, x_i, rng, i, t_steps, sqrt_recip_alphas_cumprod_steps, sqrt_recipm1_alphas_cumprod_steps, posterior_mean_coef1_steps, posterior_mean_coef2_steps,log_model_variance_steps,posterior_log_variance_clipped_steps, beta_steps,y=None,classifier_grad_fn=None,classifier_scale=None):
      """
      DDPM
      """
      if (y is not None) and (not self.class_conditional):
        assert classifier_grad_fn is not None, 'We assume that you are doing conditional generation for an unconditional model, so you must use classifier guidance'
      
      b = x_i.shape[0]
      t = batch_t(t_steps[i],b)
      sqrt_recip_alphas_cumprod = batch_t(sqrt_recip_alphas_cumprod_steps[i],b)
      sqrt_recipm1_alphas_cumprod = batch_t(sqrt_recipm1_alphas_cumprod_steps[i],b)
      posterior_mean_coef1 = batch_t(posterior_mean_coef1_steps[i],b)
      posterior_mean_coef2 = batch_t(posterior_mean_coef2_steps[i],b)
      posterior_log_variance_clipped = batch_t(posterior_log_variance_clipped_steps[i],b)
      betas = batch_t(beta_steps[i],b)
      
      
      if self.learn_var:
        eps_pred, model_var_values = self.forward_prediction_function(x_i, t, train=False,y=y)
        min_log = posterior_log_variance_clipped
        max_log = jnp.log(betas)
        frac = (model_var_values + 1) / 2 # shape as x
        log_model_variance = batch_mul(frac, max_log)+ batch_mul((1 - frac), min_log)
      else:
        eps_pred = self.forward_prediction_function(x_i, t, train=False,y=y)
        log_model_variance = batch_t(log_model_variance_steps[i],b)
      
      # get x_start from eps
      x_start = batch_mul(sqrt_recip_alphas_cumprod, x_i) - batch_mul(sqrt_recipm1_alphas_cumprod, eps_pred)
      
      if self.sample_clip_denoised:
          x_start = jnp.clip(x_start, -1, 1)
          
      model_mean = batch_mul(posterior_mean_coef1, x_start) + batch_mul(posterior_mean_coef2, x_i) # 这里写错了就高兴
      
      #### Classifier Guidance ####
      if classifier_grad_fn is not None:
        # assert False, 'classifier scale is: {s}'.format(s=classifier_scale)
        model_mean = model_mean + classifier_scale * jax.lax.stop_gradient(classifier_grad_fn(x_start, y, t)) * jnp.exp(log_model_variance)
        # assert False, '不高兴的'
      
      #### END ####

      noise = jax.random.normal(rng, x_i.shape)

      nonzero_mask = jnp.where(t > 0.5, 1, 0) # no noise when t == 0
      sample = model_mean + batch_mul(batch_mul(nonzero_mask, noise) , jnp.exp(0.5 * log_model_variance))

      # return sample, x_start # for debug
      # return x_start
      return sample
  
  def sample_one_step_DDIM(self, x_i, rng, i, t_steps, alpha_cumprod_steps, alpha_cumprod_prev_steps,y=None):
    """
    rng here is useless, if we set eta = 0
    """
    # we only implement 'generalized' here
    # we only implement 'skip_type=uniform' here
    b = x_i.shape[0]
    t = batch_t(t_steps[i],b)
    at = batch_t(alpha_cumprod_steps[i],b)
    at_next = batch_t(alpha_cumprod_prev_steps[i],b)

    assert not self.learn_var, 'DDIM only supports fixed variance DDPMs'
    eps = self.forward_prediction_function(x_i, t, train=False)
    # x0_t = (x_i - eps * jnp.sqrt(1 - at)) / jnp.sqrt(at)
    x0_t = batch_mul(x_i - batch_mul(eps, jnp.sqrt(1 - at)), 1. / jnp.sqrt(at))  # when eta=0, no need to add noise
    # when eta=0, no need to add noise
    if self.sample_clip_denoised:
      x0_t = jnp.clip(x0_t, -1, 1)
    
    c2 = jnp.sqrt(1 - at_next)
    # x_next = jnp.sqrt(at_next) * x0_t + c2 * eps
    x_next = batch_mul(x0_t, jnp.sqrt(at_next)) + batch_mul(eps, c2)
    return x_next
    # x_next = x0_t = x_i
    # print(at, at_next) # debug
    # return x_next, x0_t # debug

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

  def forward_prediction_function(self, z, t, augment_label=None,y=None, train: bool = True):  # EDM

    if not self.class_conditional:
      y = None
    t_cond = self.t_preprocess_fn(t).astype(self.dtype)
    u_pred = self.net(z, t_cond, augment_label=augment_label, train=train, y=y)
    if self.learn_var:
      return jnp.split(u_pred, 2, axis=-1)
    return u_pred

  # def forward_DDIM_pred_function(self, z, t, augment_label=None, train: bool = True):  # DDIM
  #   t_cond = jnp.zeros_like(t) if self.no_condition_t else t
  #   eps_pred = self.net(z, t_cond, augment_label=augment_label, train=train)
  #   return eps_pred
  
  def forward_FM(self, imgs, labels, augment_label, noise_batch, t_batch, train: bool = True):
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
    u_pred = self.forward_prediction_function(z, t)


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
  
  def forward_Diffusion(self, imgs, labels, augment_label, noise_batch, t_batch, train: bool = True):
    """
    You should first sample the noise and t and input them
    """
    o = self.training_diffusion_schedule()
    o = index_dictionary(o, t_batch)
    alpha_cumprod_batch, alpha_cumprod_prev_batch, posterior_log_variance_clipped_batch, beta_batch = o['alphas_cumprod'], o['alphas_cumprod_prev'], o['posterior_log_variance_clipped'], o['betas']
    # TODO: write here
    
    imgs = imgs.astype(self.dtype)
    gt = imgs
    x = imgs
    bz = imgs.shape[0]

    assert noise_batch.shape == x.shape
    assert t_batch.shape == (bz,)
    assert (labels is None) or labels.shape == (bz,)
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
    v_target = x_prior
    # v_target = jnp.ones_like(x_data)  # dummy


    # forward network
    if self.learn_var:
      u_pred, model_var_output = self.forward_prediction_function(x_mixtue, t,y=labels)
    else:
      u_pred = self.forward_prediction_function(x_mixtue, t,y=labels)

    # loss
    loss = (v_target - u_pred)**2
        
    if self.average_loss:
      loss = jnp.mean(loss, axis=(1, 2, 3))  # mean over pixels
    else:
      loss = jnp.sum(loss, axis=(1, 2, 3))  # sum over pixels
    
    if self.learn_var:
        assert self.average_loss # incompatible scale
        ####  if learn var, then have a VLB loss ####
        
        ## ------------------ constants ----------------------
        alphas_cumprod_prev = alpha_cumprod_prev_batch
        posterior_mean_coef1 = betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * jnp.sqrt(1 - betas) / (1.0 - alphas_cumprod)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = posterior_log_variance_clipped_batch
        sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod - 1.0)
        
        ## ------------------ GT values -------------------
        posterior_mean = batch_mul(posterior_mean_coef1, x_data) + batch_mul(posterior_mean_coef2, x_mixtue)
        true_mean = posterior_mean
        true_log_variance_clipped = posterior_log_variance_clipped # these two has no rely on model
        
        ## ------------------ Model values -------------------
        model_output, model_var_values = jax.lax.stop_gradient(u_pred), model_var_output # follow the paper: stop grad at here
        min_log = posterior_log_variance_clipped
        max_log = jnp.log(betas)
        frac = (model_var_values + 1) / 2 # shape as x
        model_log_variance = batch_mul(frac, max_log)+ batch_mul((1 - frac), min_log)
        
        eps_prediction = model_output
        x_start = batch_mul(sqrt_recip_alphas_cumprod, x_mixtue) - batch_mul(sqrt_recipm1_alphas_cumprod, eps_prediction)
        if self.sample_clip_denoised:
          x_start = jnp.clip(x_start, -1, 1)
        model_mean = batch_mul(posterior_mean_coef1, x_start) + batch_mul(posterior_mean_coef2, x_mixtue)
        
        ## --------------------- calculate VLB ------------------
        # KL divergence
        assert true_mean.shape == model_mean.shape == model_log_variance.shape == x_data.shape, 'true_mean shape: {t}, model_mean shape: {m}, x_data shape: {x}, model_log_variance shape: {v}'.format(t=true_mean.shape, m=model_mean.shape, x=x_data.shape, v=model_log_variance.shape)
        
        true_log_variance_clipped_shape_as_x = true_log_variance_clipped.reshape(bz, 1, 1, 1)
        
        kld = 0.5 * (
          -1.0 # broadcast
          + model_log_variance # shape as x
          - true_log_variance_clipped_shape_as_x # shape as x
          + jnp.exp(true_log_variance_clipped_shape_as_x - model_log_variance) # shape as x
          + (model_mean - true_mean)**2 * jnp.exp(-model_log_variance) # shape as x
        ) # kld has same shape as x
        kld = kld.mean(axis=(1,2,3)) / jnp.log(2.0)
        
        # corner case: when t=0, becomes NLL
        decoder_nll = - discretized_gaussian_log_likelihood(x=x_data, means=model_mean, log_scales=0.5 *model_log_variance)
        assert decoder_nll.shape == x_data.shape
        decoder_nll = decoder_nll.mean(axis=(1, 2, 3)) / jnp.log(2.0)  # mean over pixels
        
        VLB = jnp.where((t < 0.5), decoder_nll, kld)
      
        ## --------------------- Finally done! -----------------
        loss = loss + VLB  # add VLB loss

    loss = loss.mean()  # mean over batch
    loss_train = loss

    dict_losses = {}
    dict_losses['loss'] = loss  # legacy
    dict_losses['loss_train'] = loss_train

    # prepare some visualization
    # if we can pred u, then we can reconstruct x_data from x_prior
    # x_data_pred = z + batch_mul(t, u_pred)
    # x_data_pred = z + batch_mul(1 - t, u_pred)
    sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod - 1.0)
    x_data_pred = batch_mul(sqrt_recip_alphas_cumprod, x_mixtue) - batch_mul(sqrt_recipm1_alphas_cumprod, u_pred)
    # x_data_sanity = batch_mul(sqrt_recip_alphas_cumprod, x_mixtue) - batch_mul(sqrt_recipm1_alphas_cumprod, v_target)
    

    images = self.get_visualization(
      [gt,        # image (from dataset)
       v_target,  # target of network (known)
       u_pred,    # prediction of network
       x_mixtue,         # input to network (noisy image)
       x_data_pred, # prediction of clean image, reparameterized by the network
      # x_data_sanity, # sanity check, this should be the same as `gt`
      ])

    return loss_train, dict_losses, images
  
  def forward_edm_denoising_function(self, x, sigma, augment_label=None, train: bool = True):  # EDM
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

    c_in = 1 / jnp.sqrt(sigma ** 2 + self.data_std ** 2)

    # forward network
    in_x = batch_mul(x, c_in)
    sigma = sigma.reshape(sigma.shape[0])

    # F_x = self.net(in_x, c_noise, augment_label=augment_label, train=train)
    F_x = self.forward_prediction_function(in_x, sigma, augment_label=augment_label, train=train)

    D_x = batch_mul(x, c_skip) + batch_mul(F_x, c_out)
    return D_x
  
  def forward_EDM(self, imgs, labels, augment_label, noise_batch, t_batch, train: bool = True):
    """
    code from edm
    for FM API use
    ---
    input: x (noisy image, =x+sigma*noise), sigma (condition)
    We hope this function operates D(x+sigma*noise) = x
    our network has F((1-t)x + t*noise) = x - noise
    """
    
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
    D_xn = self.forward_edm_denoising_function(xn, sigma, augment_label)

    # loss
    loss = (D_xn - gt)**2
    loss = batch_mul(loss, weight)

    if self.average_loss:
      raise ValueError("we recommend to use sum loss")
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
    images = self.get_visualization([gt, xn, D_xn])

    return loss_train, dict_losses, images
  
  def forward_Classifier(self, imgs, labels, augment_label, noise_batch, t_batch, train: bool = True):
    # 别被忽悠了: 这个train没有用！
    
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
    o = self.training_diffusion_schedule()
    o = index_dictionary(o, t_batch)
    alpha_cumprod_batch, alpha_cumprod_prev_batch, posterior_log_variance_clipped_batch, beta_batch = o['alphas_cumprod'], o['alphas_cumprod_prev'], o['posterior_log_variance_clipped'], o['betas']
    
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
  
  def pre_process_for_t_prediction_Diffusion(self, imgs, labels, augment_label, noise_batch, t_batch, train: bool = True):
    imgs = imgs.astype(self.dtype)
    x = imgs
    bz = imgs.shape[0]

    # -----------------------------------------------------------------
    #  diffusion alpha, betas
    o = self.training_diffusion_schedule()
    o = index_dictionary(o, t_batch)
    alpha_cumprod_batch, alpha_cumprod_prev_batch, posterior_log_variance_clipped_batch, beta_batch = o['alphas_cumprod'], o['alphas_cumprod_prev'], o['posterior_log_variance_clipped'], o['betas']
    
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
    
    # we normalize t into mean 0 and std 1, before letting our network to predict
    # for DDPM: 0, 1, 2, ..., self.diffusion_nT - 1 -> approx as [0, self.diffusion_nT]
    t_batch = t_batch / self.diffusion_nT # [0, 1]
    t_batch = (t_batch - 0.5) * jnp.sqrt(jnp.array(12.0, dtype=jnp.float32))  # std becomes 1
    
    return x_mixtue, t_batch
  
  def pre_process_for_t_prediction_FM(self, imgs, labels, augment_label, noise_batch, t_batch, train: bool = True):
    imgs = imgs.astype(self.dtype)
    x = imgs
    bz = imgs.shape[0]


    # sample from data
    x_data = x
    # sample from prior (noise)
    x_prior = noise_batch

    t = t_batch
    t = t * (1 - self.eps) + self.eps
    x_mixtue = batch_mul(1-t, x_data) + batch_mul(t, x_prior)
    # x_mixtue = batch_mul(t, x_data) + batch_mul(1-t, x_prior)
    
    # we normalize t into mean 0 and std 1, before letting our network to predict
    # for FM: [self.eps,1], 但是因为我想写，所以就当[0,1]
    # t = (t - 0.5) * jnp.sqrt(jnp.array(12.0, dtype=jnp.float32))  # std becomes 1
    
    return x_mixtue, t
  
  def forward_T_predictor_main(self, pre_process_fn, imgs, labels, augment_label, noise_batch, t_batch, train: bool = True):
    
    bz = imgs.shape[0]
    assert noise_batch.shape == imgs.shape
    assert t_batch.shape == (bz,)
    
    # x_mixtue, v_target = pre_process_fn(imgs, labels, augment_label, noise_batch, t_batch, train=train)
    # x_mixtue = noise_batch
    # x_mixtue = batch_mul(t_batch, imgs) + batch_mul(1-t_batch, noise_batch)
    x_mixtue = batch_mul(1-t_batch, imgs) + batch_mul(t_batch, noise_batch)
    v_target = t_batch

    # forward network
    u_pred = self.forward_prediction_function(x_mixtue, t_batch, train=train,y=labels).reshape((bz,) ) # shape: [b, 1] -> [b]

    # loss: MSE loss
    assert v_target.shape == u_pred.shape, 'v_target shape: {v}, u_pred shape: {u}'.format(v=v_target.shape, u=u_pred.shape)
    assert v_target.shape == (bz,), 'v_target shape: {v}'.format(v=v_target.shape)
    loss = jnp.mean((v_target - u_pred)**2)

    loss_train = loss
    
    # vis & log
    vis_target = v_target.reshape(bz, 1, 1, 1).repeat(32, axis=1).repeat(32,axis=2).repeat(3,axis=3).astype(jnp.float32)
    vis_pred = u_pred.reshape(bz, 1, 1, 1).repeat(32, axis=1).repeat(32,axis=2).repeat(3,axis=3).astype(jnp.float32)
    err = jnp.abs(v_target - u_pred).reshape(bz, 1, 1, 1).repeat(32, axis=1).repeat(32,axis=2).repeat(3,axis=3).astype(jnp.float32) * 25

    dict_losses = {}
    dict_losses['loss'] = loss  # legacy
    dict_losses['loss_train'] = loss_train
    dict_losses['acc_train'] = jnp.mean((jnp.abs(v_target - u_pred) < 1e-2).astype(jnp.float32))
    dict_losses['pred_mean'] = jnp.mean(u_pred)
    dict_losses['target_mean'] = jnp.mean(v_target)
    dict_losses['pred_std'] = jnp.std(u_pred)
    dict_losses['target_std'] = jnp.std(v_target)

    images = self.get_visualization(
      [imgs,        # image (from dataset)
       x_mixtue,  # mixture of data and noise
        vis_target,  # target of network (known)
        vis_pred,    # prediction of network
        err,   # error, black is good
      ])

    return loss_train, dict_losses, images
  
  def forward(self, *args, **kwargs):
      if self.task == 'FM':
        return self.forward_FM(*args, **kwargs)
      elif self.task == 'Diffusion':
        return self.forward_Diffusion(*args, **kwargs)
      elif self.task == 'Classifier':
        return self.forward_Classifier(*args, **kwargs)
      elif self.task in ['EDM', 'edm']:
        return self.forward_EDM(*args, **kwargs)
      elif 't_predictor' in self.task:
        if self.task == 'FM_t_predictor':
          return self.forward_T_predictor_main(self.pre_process_for_t_prediction_FM, *args, **kwargs)
        elif self.task == 'Diffusion_t_predictor':
          return self.forward_T_predictor_main(self.pre_process_for_t_prediction_Diffusion, *args, **kwargs)
        else:
          raise NotImplementedError(f'Unknown t predicting task: {self.task}')
      else:
        raise NotImplementedError(f'Unknown task: {self.task}')
  
  def __call__(self, imgs, labels, train: bool = False):
    # initialization only
    t = jnp.ones((imgs.shape[0],))
    augment_label = jnp.ones((imgs.shape[0], 9)) if self.use_aug_label else None  # fixed augment_dim # TODO: what is this?
    out = self.net(imgs, t, augment_label,y=jnp.ones((imgs.shape[0],))) # TODO: whether to add train=train
    out_ema = None   # no need to initialize it here
    return out, out_ema
