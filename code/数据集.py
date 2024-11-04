import torch
import random
import numpy as np
import jax
from absl import logging
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from functools import partial

def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def create_split(
    dataset,
    batch_size,
    split,
    config,
):
  """Creates a split from the ImageNet dataset using Torchvision Datasets.

  Args:
    config: Configurations for the dataset.
    batch_size: Batch size for the dataloader.
    split: 'train' or 'val'.
  Returns:
    it: A PyTorch Dataloader.
    steps_per_epoch: Number of steps to loop through the DataLoader.
  """
  rank = jax.process_index()
  if split == 'train':
    if rank == 0:
      logging.info(dataset)
    # sqa's copy from deit's sampler, which implements the RASampler

    sampler = DistributedSampler(
      dataset,
      num_replicas=jax.process_count(),
      rank=rank,
      shuffle=True,
    )
    it = DataLoader(
      dataset, batch_size=batch_size, drop_last=True,
      worker_init_fn=partial(worker_init_fn, rank=rank),
      sampler=sampler,
      num_workers=config.num_workers,
      prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
      pin_memory=True,
      persistent_workers=True if config.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)

  elif split == 'val':
    logging.info(dataset)

    sampler = DistributedSampler(
      dataset,
      num_replicas=jax.process_count(),
      rank=rank,
      shuffle=False,  # don't shuffle for val
    )
    it = DataLoader(
      dataset,
      batch_size=batch_size,
      drop_last=False,
      worker_init_fn=partial(worker_init_fn, rank=rank),
      sampler=sampler,
      num_workers=config.num_workers,
      prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
      pin_memory=True,
      persistent_workers=True if config.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)
  else:
    raise NotImplementedError

  return it, steps_per_epoch

def prepare_batch_data_sqa(image):

  local_device_count = jax.local_device_count()
  image = image.permute(0, 2, 3, 1)
  image = image.reshape((local_device_count, -1) + image.shape[1:])

  image = image.numpy()

  return image