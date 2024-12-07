# Your configurations here
source config.sh
CONDA_ENV=$OWN_CONDA_ENV_NAME

USE_FAKE_DATASET=1
# USE_FAKE_DATASET=2

############# No need to modify #############

if [[ $USE_FAKE_DATASET == 1 ]]; then
    export USE_DATA_ROOT=$FAKE_DATA_ROOT
else
    export USE_DATA_ROOT=/$DATA_ROOT/data/imagenet
fi

echo 'Using data root: '$USE_DATA_ROOT

source ka.sh

echo Running at $VM_NAME $ZONE

STAGEDIR=/$DATA_ROOT/staging/$(whoami)/debug-$VM_NAME
sudo mkdir -p $STAGEDIR
sudo chmod 777 -R $STAGEDIR
echo 'Staging files...'
sudo rsync -a . $STAGEDIR --exclude=tmp --exclude=.git --exclude=__pycache__ --exclude='*.png' --exclude=wandb
echo 'staging dir: '$STAGEDIR
echo 'Done staging.'

LOGDIR=$STAGEDIR/log
sudo rm -rf $LOGDIR
sudo mkdir -p ${LOGDIR}
sudo chmod 777 -R ${LOGDIR}
echo 'Log dir: '$LOGDIR


if [[ $USE_CONDA == 1 ]]; then
    CONDA_PATH=$(which conda)
    CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
fi

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR
echo 'Current dir: '
pwd
if [ \"$USE_CONDA\" -eq 1 ]; then
    echo 'Using conda'
    source $CONDA_INIT_SH_PATH
    conda activate $CONDA_ENV
fi
which python
which pip3
python3 main.py \
    --workdir=${LOGDIR} \
    --mode=remote_debug \
    --config=configs/load_config.py:remote_debug \
" 2>&1 | tee -a $LOGDIR/output.log

############# No need to modify [END] #############