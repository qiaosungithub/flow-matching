import jax
jax.distributed.initialize()
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

"""Main file for running the ImageNet example.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

import os
from absl import app
from absl import flags
from absl import logging
# from clu import platform
import jax
from ml_collections import config_flags

import train_t
from utils import logging_util
from utils.logging_util import log_for_0
logging_util.supress_checkpt_info()

import warnings
warnings.filterwarnings("ignore")


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_bool('debug', False, 'Debugging mode.')
flags.DEFINE_enum('mode', enum_values=['local_debug','remote_debug','remote_run'], default='remote_run', help='Running mode.') # NOTE: This variable isn't used currently, but maintained for future use. This at least ensures that there is no more variable that must be passed in from the command line.

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  log_for_0('JAX process: %d / %d', jax.process_index(), jax.process_count())
  log_for_0('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  # platform.work_unit().set_task_status(
  #     f'process_index: {jax.process_index()}, '
  #     f'process_count: {jax.process_count()}'
  # )
  # platform.work_unit().create_artifact(
  #     platform.ArtifactType.DIRECTORY, FLAGS.workdir, 'workdir'
  # )

  log_for_0('FLAGS.config: \n{}'.format(FLAGS.config))

  if FLAGS.config.load_from is not None:
    train_t.just_evaluate(FLAGS.config, FLAGS.workdir)
  elif FLAGS.debug:
    with jax.disable_jit():
      train_t.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  else:
    train_t.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  # train_t.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  # logging_util.verbose_off()
  # logging_util.set_time_logging(logging)
  flags.mark_flags_as_required(['config', 'mode', 'workdir'])
  app.run(main)