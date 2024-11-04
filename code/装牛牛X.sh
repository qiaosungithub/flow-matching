conda create -n NNX python==3.10.14 -y
conda activate NNX # These two lines are very smart. If on a device there is no conda, then these two lines error out, but the remaining can still be run.
pip install jax[tpu]==0.4.27 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install jaxlib==0.4.27 "flax>=0.8"
pip install -r requirements.txt # other tang dependencies