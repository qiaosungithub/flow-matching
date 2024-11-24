# from absl import logging
# import jax
# from flax.training import checkpoints
# from utils.logging_util import log_for_0
# import flax.nnx as nn

# def restore_checkpoint(state, workdir):
#   return checkpoints.restore_checkpoint(workdir, state)

# # def 变(p, d: dict):
# #     for k in p:
# #         v = p[k]
# #         # print(type(v))
# #         if isinstance(v, nn.variablelib.VariableState):
# #             d[str(k)] = type(v)
# #         else:
# #             newd={}
# #             d[str(k)]=newd
# #             变(v, newd)
# def save_checkpoint(state, workdir):
#   state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
#   step = int(state.step)
#   log_for_0('Saving checkpoint step %d.', step)
#   print("In saving checkpoint", flush=True)
#   # print(state, flush=True)
#   # d = {}
#   # 变(state, d)
#   # print(d, flush=True)
#   checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=10)


# def restore_pretrained(state, path, config):
#   pretrained = checkpoints.restore_checkpoint(path, target=None)
#   log_for_0(f'pretrained model: {pretrained.keys()}')

#   assert jax.tree_structure(state.params['Encoder']) == \
#     jax.tree_structure(pretrained['params']['Encoder'])
#   assert jax.tree_structure(state.params['Decoder']) == \
#     jax.tree_structure(pretrained['params']['Decoder'])

#   state.params['Encoder'] = pretrained['params']['Encoder']
#   state.params['Decoder'] = pretrained['params']['Decoder']

#   # just in case
#   # assert jax.tree_structure(state.batch_stats) == \
#   #   jax.tree_structure(pretrained['batch_stats'])
#   # state = state.replace(batch_stats=pretrained['batch_stats'])

#   logging.info('Loaded.')
#   return state
raise NotImplementedError

