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
from typing import Any

from flax import jax_utils as ju
from flax.training.train_state import TrainState as FlaxTrainState
from flax.training import checkpoints
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import optax
import torch
import numpy as np
import flax.nnx as nn

from utils.info_util import print_params
from utils.display_utils import show_dict, count_params

import models.t.t as t

NUM_CLASSES = 10

class NNXTrainState(FlaxTrainState):
  batch_stats: Any
  rng_states: Any
  graphdef: Any
  useless_variable_state: Any
  # NOTE: is_training can't be a attr, since it can't be replicated


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

def restore_checkpoint(model_init_fn, state, workdir, model_config):
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
    'you_hua_qi': state.opt_state,
    'step': 0
  }
  loaded_state = checkpoints.restore_checkpoint(workdir, target=fake_state,orbax_checkpointer=checkpointer)
  merged_params = loaded_state['mo_xing']
  opt_state = loaded_state['you_hua_qi']
  step = loaded_state['step']
  params, batch_stats, _ = merged_params.split(nn.Param, nn.BatchStat, nn.VariableState)
  return state.replace(
    params=params,
    rng_states=rng_states,
    batch_stats=batch_stats,
    opt_state=opt_state,
    step=step
  )

def create_train_state(
  model, image_size, learning_rate_fn
):
  """
  Create initial training state, including the model and optimizer.
  config: the training config
  ---
  this version is for eval
  apply_fn takes noisy image and output t ((1-t)x+t eps)
  """
  # print("here we are in the function 'create_train_state' in train.py; ready to define optimizer")
  graphdef, params, batch_stats, rng_states, useless_variable_states = nn.split(model, nn.Param, nn.BatchStat, nn.RngState, nn.VariableState)

  print_params(params)

  def apply_fn(graphdef2, params2, rng_states2, batch_stats2, useless_, is_training, noisy_image):

    merged_model = nn.merge(graphdef2, params2, rng_states2, batch_stats2, useless_)
    if is_training:
      merged_model.train()
    else:
      merged_model.eval()
    del params2, rng_states2, batch_stats2, useless_
    t_pred = merged_model.forward(noisy_image)
    # for debug
    # t_pred = merged_model.forward(jnp.ones_like(noise_batch) * t)
    new_batch_stats, new_rng_states, _ = nn.state(merged_model, nn.BatchStat, nn.RngState, ...)
    return t_pred, new_batch_stats, new_rng_states

  # here is the optimizer
  tx = optax.adamw(
    learning_rate=learning_rate_fn,
    b1=0.9,
    b2=0.999,
    weight_decay=0,
  )
  
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

def init_t_network(debug=False):
  """
  debug: whether to replicate the state
  """
  model_cls = t.sqa_t_ver1
  rngs = nn.Rngs(0)
  model = model_cls(rngs=rngs)
  show_dict(f'number of model parameters:{count_params(model)}')

  ########### Create LR FN ###########
  learning_rate_fn = lambda:1 # just in order to create the state

  ########### Create Train State ###########
  state = create_train_state(model, 32, learning_rate_fn)
  # assert config.get('load_from',None) is not None, 'Must provide a checkpoint path for evaluation'
  # if not os.path.isabs(config.load_from):
  #   raise ValueError('Checkpoint path must be absolute')
  # if not os.path.exists(config.load_from):
  #   raise ValueError('Checkpoint path {} does not exist'.format(config.load_from))
  state = restore_checkpoint(model_cls, state, "/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20241128_031750_8xab8k_kmh-tpuvm-v2-32-preemptible-2__b_lr_ep_eval/checkpoint_4850", {}) # change the path to the checkpoint path here !!!!!!!!!
  state_step = int(state.step)
  if not debug:
    state = ju.replicate(state) # NOTE: this doesn't split the RNGs automatically, but it is an intended behavior
  return state
