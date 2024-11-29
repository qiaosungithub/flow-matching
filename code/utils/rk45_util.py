import jax
import jax.numpy as jnp
from jax import random
import flax.nnx as nn
from utils.info_util import log_for_0

from functools import partial

from absl import logging

import numpy as np
import time
from scipy import integrate

def get_rk45_functions(model, config, rng):

  def flow_step(state, x, t):
    merged_model = nn.merge(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state)
    u_pred = merged_model.forward_flow_pred_function(x, t, train=False)
    return u_pred

  p_flow_step = jax.pmap(
    flow_step,
    axis_name='batch',
  )

  image_size = config.model.image_size

  # x_fake = jnp.ones((jax.local_device_count(), config.fid.device_batch_size, image_size, image_size, config.model.out_channels), jnp.float32)
  # t_fake = jnp.ones((jax.local_device_count(), config.fid.device_batch_size), jnp.float32)
  # lowered = p_flow_step.lower(
  #   params={'params': {'net': state.params['net']}, 'batch_stats': {}},
  #   x=x_fake,
  #   t=t_fake,
  # )
  # logging.info('Compiling p_flow_step...')
  # t_start = time.time()
  # p_flow_step = lowered.compile()
  # logging.info('p_flow_step compiled in {}s'.format(time.time() - t_start))
  # out = p_flow_step(params={'params': {'net': state.params['net']}, 'batch_stats': {}}, x=x_fake, t=t_fake)[0]
  p_sample_step = p_flow_step  # rename for legacy

  rng_init = rng

  def run_p_sample_step(p_sample_step, state, sample_idx):
    # this is the ode solver for one stage
    def ode_func(t, x):
      x = jnp.array(x)
      x = x.reshape((jax.local_device_count(), -1, image_size, image_size, config.model.out_channels))
      # t = t.reshape((jax.local_device_count(), -1))
      t = jnp.ones(x.shape[:2], jnp.float32) * t
      out = p_sample_step(state, x, t)
      # print('out.shape:', out.shape)
      jax.random.normal(jax.random.key(0), ()).block_until_ready()
      out = np.array(out)
      out = out.reshape((-1,))
      # print('out.shape:', out.shape)
      return out

    x_shape = (jax.local_device_count(), config.fid.device_batch_size, image_size, image_size, config.model.out_channels)
    # print('x_shape:', x_shape)

    x_init = []
    for i in sample_idx: # here we fold in sample_idx to each device
      rng_i = random.fold_in(rng_init, i)
      x_init.append(jax.random.normal(rng_i, x_shape[1:], jnp.float32))
    x_init = jnp.stack(x_init, axis=0)  # [8, b, 32, 32, 3]
    # print('x_init.shape:', x_init.shape)
    x = np.array(x_init).flatten()

    # Black-box ODE solver for the probability flow ODE
    rtol = atol = 1e-4
    solution = integrate.solve_ivp(ode_func, (1e-3, 1.0), x, rtol=rtol, atol=atol, method='RK45')
    x = solution.y[:, -1]
    x = x.reshape((jax.local_device_count() * config.fid.device_batch_size, image_size, image_size, config.model.out_channels))
    images = x

    nfe = solution.nfev
    log_for_0('nfe: {}'.format(nfe))

    return images

  return run_p_sample_step, p_sample_step

# def get_rk45_functions(model, config, state, rng):
#   def flow_step(model, params, x, t):
#     u_pred = model.apply(
#         params,  # which is {'params': state.params, 'batch_stats': state.batch_stats},
#         x, t,
#         rngs={},
#         train=False,
#         method=model.forward_flow_pred_function,
#         mutable=['batch_stats'],
#     )
#     return u_pred

#   p_flow_step = jax.pmap(
#     functools.partial(flow_step, model=model,),
#     axis_name='batch',
#   )

#   image_size = config.model.image_size

#   x_fake = jnp.ones((jax.local_device_count(), config.fid.device_batch_size, image_size, image_size, config.model.out_channels), jnp.float32)
#   t_fake = jnp.ones((jax.local_device_count(), config.fid.device_batch_size), jnp.float32)
#   lowered = p_flow_step.lower(
#     params={'params': {'net': state.params['net']}, 'batch_stats': {}},
#     x=x_fake,
#     t=t_fake,
#   )
#   logging.info('Compiling p_flow_step...')
#   t_start = time.time()
#   p_flow_step = lowered.compile()
#   logging.info('p_flow_step compiled in {}s'.format(time.time() - t_start))
#   # out = p_flow_step(params={'params': {'net': state.params['net']}, 'batch_stats': {}}, x=x_fake, t=t_fake)[0]
#   p_sample_step = p_flow_step  # rename for legacy

#   rng_init = rng
#   def run_p_sample_step(p_sample_step, state, sample_idx, ema=False):
#     # this is the ode solver for one stage
#     net_key = 'net_ema' if ema else 'net'
#     def ode_func(t, x):
#       x = jnp.array(x)
#       x = x.reshape((jax.local_device_count(), -1, image_size, image_size, config.model.out_channels))
#       # t = t.reshape((jax.local_device_count(), -1))
#       t = jnp.ones(x.shape[:2], jnp.float32) * t
#       out = p_sample_step(params={'params': {'net': state.params[net_key]}, 'batch_stats': {}}, x=x, t=t)[0]
#       jax.random.normal(jax.random.key(0), ()).block_until_ready()
#       out = np.array(out)
#       out = out.reshape((-1,))
#       return out

#     x_shape = (jax.local_device_count(), config.fid.device_batch_size, image_size, image_size, config.model.out_channels)

#     x_init = []
#     for i in sample_idx:
#       rng_i = random.fold_in(rng_init, i)
#       x_init.append(jax.random.normal(rng_i, x_shape[1:], jnp.float32))
#     x_init = jnp.stack(x_init, axis=0)  # [4, b, 32, 32, 3]
#     x = np.array(x_init).flatten()

#     # Black-box ODE solver for the probability flow ODE
#     rtol = atol = 1e-4
#     solution = integrate.solve_ivp(ode_func, (1e-3, 1.0), x, rtol=rtol, atol=atol, method='RK45')
#     x = solution.y[:, -1]
#     x = x.reshape((jax.local_device_count() * config.fid.device_batch_size, image_size, image_size, config.model.out_channels))
#     images = x

#     nfe = solution.nfev
#     logging.info('nfe: {}'.format(nfe))

#     return images

#   return run_p_sample_step, p_sample_step