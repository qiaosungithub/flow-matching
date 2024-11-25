# Run job in a remote TPU VM
source ka.sh # import VM_NAME, ZONE

echo $VM_NAME $ZONE

CONDA_ENV=$OWN_CONDA_ENV_NAME

if [[ $USE_CONDA == 1 ]]; then
    CONDA_PATH=$(which conda)
    CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
fi

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
if [ \"$USE_CONDA\" -eq 1 ]; then
    echo 'Using conda'
    source $CONDA_INIT_SH_PATH
    conda activate $CONDA_ENV
fi
which python3
which pip3
python3 -m wandb login $WANDB_API_KEY
sleep 1
python3 -m wandb login
"