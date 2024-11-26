# initialize and set up remote TPU VM
source ka.sh # import VM_NAME, ZONE

echo $VM_NAME $ZONE


gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
if [ \"$USE_CONDA\" -eq 1 ]; then
    echo 'Using conda'
    source $CONDA_INIT_SH_PATH
    conda activate $CONDA_ENV
fi
which python3
which pip3
python3 -c 'import jax; print(jax.devices())'
python3 -c 'import flax.nnx as nn; print(nn.Linear)'
"

# pip install wandb