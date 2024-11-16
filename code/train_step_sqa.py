from jax import random, lax
from typing import Any
import jax
import jax.numpy as jnp
from functools import partial
from flax.training.train_state import TrainState as FlaxTrainState

class NNXTrainState(FlaxTrainState):
  batch_stats: Any
  rng_states: Any
  graphdef: Any
  useless_variable_state: Any
  # NOTE: is_training can't be a attr, since it can't be replicated

def criterion(pred, target):
  # L2 lossp
  # return jnp.mean((pred - target) ** 2, axis=0).mean()
  return jnp.mean((pred - target) ** 2, axis=0).sum()

def train_step_compute(state:NNXTrainState, arg_batch, t_batch, target_batch):

  def loss_fn(real_params):
    preds, new_batch_stats, new_rng_params = state.apply_fn(state.graphdef, real_params, state.rng_states, state.batch_stats, state.useless_variable_state, True, arg_batch, t_batch) # True: is_training
    loss = criterion(preds, target_batch)
    # customized weight decay (don't apply to bias)
    return loss, (new_batch_stats, new_rng_params)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  grads = lax.pmean(grads, axis_name='batch')

  new_batch_stats, new_rng_params = aux[1]

  loss = aux[0]
  loss = lax.pmean(loss, axis_name='batch')
  # TODO: implement ema
  metrics = {"loss": loss}

  new_state = state.apply_gradients(
    grads=grads, batch_stats=new_batch_stats, rng_states=new_rng_params
  )

  return new_state, metrics

from jax.experimental import ode as O
def solve_diffeq(init_x,train_state:NNXTrainState,see_steps:int=10):
  def f(x, t):
    # assert t.shape == (), ValueError(f't shape: {t.shape}')
    creation = t.reshape(1,).repeat(x.shape[0],axis=0)
    out = train_state.apply_fn(train_state.graphdef, train_state.params, train_state.rng_states, train_state.batch_stats, train_state.useless_variable_state,False, x,creation)[0]
    return out
  out = O.odeint(f, init_x, jnp.linspace(0,1,see_steps), rtol=1e-5, atol=1e-5) # (8, 32, 32, 3)
  # out = lax.all_gather(out, axis_name='batch')
  # (10, 8, 32, 32, 3)
  return out

p_solve_diffeq = jax.pmap(
  partial(solve_diffeq,see_steps=10),
  axis_name='batch') # default: see_steps=10

p_solve_diffeq_only_out = jax.pmap(
  partial(solve_diffeq,see_steps=2),
  axis_name='batch')

p_train_step = jax.pmap(train_step_compute, axis_name='batch')

def sample_for_fid(state:NNXTrainState, sample_idx, rng_init, device_bs, image_size):
  """
  We will pmap this function
  ---
  input: 
    sample_idx: use different sample_idx to get different samples
  ---
  output: images with shape (n_sample, h, w, c) (where c = 3)
  """
  rng_sample = random.fold_in(rng_init, sample_idx)  # fold in sample_idx
  init_x = jax.random.normal(rng_sample, (device_bs, image_size, image_size, 3))
  samples = solve_diffeq(init_x, state, see_steps=2) # (2, device_bs, image_size, image_size, 3)
  samples = samples[1] # (device_bs, image_size, image_size, 3)
  # samples = lax.all_gather(samples, axis_name='batch') # TODO: what is the shape?
  return samples

def fast_generate(state:NNXTrainState, t_cur, inputs, delta_t):
  x_i = inputs
  v_i, _, _ = state.apply_fn(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state, False, x_i, t_cur)
  outputs = x_i + v_i * delta_t
  return outputs

p_fast_generate = jax.pmap(fast_generate, axis_name='batch')

def generate(state:NNXTrainState, dtype, n_sample, config, image_size):
  """
  Generate samples from the model
  config: sampling config
  we do not use rng
  """

  # prepare schedule
  num_replicas = jax.local_device_count()
  # num_steps = model.n_T
  num_steps = config.n_T
  step_indices = jnp.arange(num_steps, dtype=dtype)
  t_steps = step_indices / num_steps  # t_i = i / N
  t_steps = jnp.concatenate([t_steps, jnp.ones((1,), dtype=dtype)], axis=0)  # t_N = 1; no need to round_sigma

  # initialize noise
  x_shape = (n_sample, image_size, image_size, 3)
  # rng_used, rng = jax.random.split(rng, 2)
  rng = jax.random.PRNGKey(0)
  x_0 = jax.random.normal(rng, x_shape, dtype=dtype)
  x_0 = x_0.reshape((1, *x_shape)).repeat(num_replicas, axis=0)
  # shape (32, 64, 32, 32, 3)

  def step_fn(i, inputs):
    t_cur = t_steps[i] * jnp.ones((num_replicas, n_sample,), dtype=dtype)
    delta_t = t_steps[i + 1] - t_steps[i]
    delta_t = delta_t * jnp.ones((num_replicas, ), dtype=dtype)
    outputs = p_fast_generate(state, t_cur, inputs, delta_t)
    # x_i = inputs
    # v_i = state.apply_fn(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state, False, x_i, t_cur)
    # outputs = x_i + v_i * (t_steps[i + 1] - t_steps[i])
    return outputs

  # outputs = jax.lax.fori_loop(0, num_steps, step_fn, x_0)
  for i in range(num_steps):
    x_0 = step_fn(i, x_0)
  outputs = x_0[0]  # shape (n_sample, image_size, image_size, 3)
  return outputs