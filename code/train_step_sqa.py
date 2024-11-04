from jax import random, lax
from typing import Any
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState as FlaxTrainState

class NNXTrainState(FlaxTrainState):
  batch_stats: Any
  rng_states: Any
  graphdef: Any
  useless_variable_state: Any
  # NOTE: is_training can't be a attr, since it can't be replicated

def train_step_compute(state:NNXTrainState, images, sigmas, sigma_indices, noise):
  sigma_batch = sigmas[sigma_indices].reshape(-1, 1, 1, 1)
  noise = noise * sigma_batch
  images_noise = images + noise
  target = - noise / sigma_batch

  def loss_fn(params):
    """loss function used for training."""
    outputs, new_batch_stats, new_rng_params = state.apply_fn(
      state.graphdef, params, state.rng_states, state.batch_stats, state.useless_variable_state, True, images_noise, sigma_indices)
    outputs = outputs * sigma_batch
    loss = jnp.mean((outputs - target)**2)
    return loss, (outputs, new_batch_stats, new_rng_params)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = lax.pmean(grads, axis_name='batch')

  outputs, new_batch_stats, new_rng_params = aux[1]

  loss = aux[0]
  loss = lax.pmean(loss, axis_name='batch')
  # TODO: implement ema
  metrics = {"loss": loss}

  new_state = state.apply_gradients(
    grads=grads, batch_stats=new_batch_stats, rng_states=new_rng_params
  )

  return new_state, metrics

fast_train_step_compute = jax.pmap(
  train_step_compute,
  axis_name='batch',
)

def train_step_sqa(state:NNXTrainState, images, rng, sigmas):
  """Perform a single training step."""

  # ResNet has no dropout; but maintain rng_dropout for future usage
  # rng_step = random.fold_in(rng_init(), state.step)
  # rng_device = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))
  # rng, _ = random.split(rng_device)
  # print("images.shape", images.shape)

  sigma_indices = random.randint(rng(), (images.shape[0], images.shape[1]), 0, len(sigmas))

  noise = random.normal(rng(), images.shape)
  assert sigmas.shape == (len(sigmas),)
  assert sigma_indices.shape == (images.shape[0], images.shape[1])
  sigmas = sigmas.reshape(1, *sigmas.shape)
  sigmas = jnp.tile(sigmas, (images.shape[0], 1))

  new_state, metrics = fast_train_step_compute(state, images, sigmas, sigma_indices, noise)

  return new_state, metrics

# def train_step_sqa(state:NNXTrainState, images, rng, sigmas):
#   """Perform a single training step. FAKE VERSION FOR DEBUG"""
#   metrics = {"loss": 0.0}

#   return state, metrics