source config-sqa.sh
rm -rf tmp # Comment this line if you want to reload (usually not the case)

CONDA_PATH=$(which conda)
CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
LOGDIR=$(pwd)/tmp
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}

source $CONDA_INIT_SH_PATH
# remember to use your own conda environment
conda activate $OWN_CONDA_ENV_NAME
export TFDS_DATA_DIR='/kmh-nfs-ssd-eu-mount/code/hanhong/dot/tensorflow_datasets'
# export TFDS_DATA_DIR='gs://kmh-gcp/tensorflow_datasets' # this is for imagenet

echo "start running main"

python3 main-sqa.py \
    --workdir=${LOGDIR} \
    --mode=local_debug \
    --config=configs/load_config.py:local_debug_sqa \
2>&1 | grep --invert-match Could