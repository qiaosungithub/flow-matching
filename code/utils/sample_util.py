from absl import logging
import torchvision
import torch
import numpy as np
import jax.numpy as jnp
import os
import jax

from utils.vis_util import float_to_uint8
from utils.logging_util import log_for_0


def get_samples_from_dir(samples_dir, config):
  # e.g.: samples_dir = '/kmh-nfs-us-mount/logs/kaiminghe/results-edm/edm-cifar10-32x32-uncond-vp'
  ds = torchvision.datasets.ImageFolder(samples_dir, transform=torchvision.transforms.PILToTensor())
  dl = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=False, drop_last=False, num_workers=12)
  samples_all = []
  for x in dl:
    samples_all.append(x[0].numpy().transpose(0,2,3,1))
  samples_all = np.concatenate(samples_all)
  samples_all = samples_all[:config.fid.num_samples]
  return samples_all


def generate_samples_for_fid_eval(state, workdir, config, p_sample_step, run_p_sample_step):
  """
  state: NNXstate
  """
  num_steps = np.ceil(config.fid.num_samples / config.fid.device_batch_size / jax.device_count()).astype(int)
  output_dir = os.path.join(workdir, 'samples')
  os.makedirs(output_dir, exist_ok=True)
  samples_all = []
  for step in range(num_steps):
    sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())
    sample_idx = jax.device_count() * step + sample_idx
    log_for_0(f'Sampling step {step} / {num_steps}...')

    samples = run_p_sample_step(p_sample_step, state, sample_idx=sample_idx)
    # print('samples.shape:', samples.shape)
    # print(f"samples min and max: {samples.min()}, {samples.max()}")
    # exit("邓东灵")
    samples = float_to_uint8(samples)
    samples_all.append(samples)
  samples_all = np.concatenate(samples_all, axis=0)
  samples_all = samples_all[:config.fid.num_samples]
  return samples_all

