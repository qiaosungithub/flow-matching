# Your configurations here
source config.sh
CONDA_ENV=$OWN_CONDA_ENV_NAME
############# No need to modify #############
for i in {1..20}; do echo "Do you remember to use TMUX?"; done
source ka.sh

echo Running at $VM_NAME $ZONE

now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
JOBNAME=${TASKNAME}/${now}_${salt}_${VM_NAME}_${CONFIG}_b${batch}_lr${lr}_ep${ep}_eval

LOGDIR=/$DATA_ROOT/logs/$USER/$JOBNAME

sudo mkdir -p ${LOGDIR}
sudo chmod 777 -R ${LOGDIR}
echo 'Log dir: '$LOGDIR
echo 'Staging dir: '$STAGEDIR

if [[ $USE_CONDA == 1 ]]; then
    CONDA_PATH=$(which conda)
    CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
fi

command() {

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR
if [ \"$USE_CONDA\" -eq 1 ]; then
    echo 'Using conda'
    source $CONDA_INIT_SH_PATH
    conda activate $CONDA_ENV
fi
echo 'Current dir: '
pwd
which python
which pip3
export TFDS_DATA_DIR=${TFDS_DATA_DIR}
python3 main.py \
    --workdir=${LOGDIR} \
    --mode=remote_run \
    --config=configs/load_config.py:remote_run \
" 2>&1 | tee -a $LOGDIR/output.log

}

command
# for i in {1..3}; do
#     if grep -q 'Additional GRPC error information' $LOGDIR/output.log; then
#         echo 'Get GRPC error, retrying...'
#         # echo > $LOGDIR/output.log
#         mv $LOGDIR/output.log $LOGDIR/output.log.$i
#         bash re_apply.sh
#         command
#     else
#         break
#     fi
# done

############# No need to modify [END] #############