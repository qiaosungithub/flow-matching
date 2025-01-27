# Stage code and run job in a remote TPU VM

# ------------------------------------------------
# Copy all code files to staging
# ------------------------------------------------
now=`date '+%y%m%d%H%M%S'`
salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
commitid=`git show -s --format=%h`  # latest commit id; may not be exactly the same as the commit
export STAGEDIR=/kmh-nfs-ssd-eu-mount/staging/sqa/debug-km-code/${now}-${salt}-${commitid}-code
sudo mkdir -p $STAGEDIR
sudo chmod 777 -R $STAGEDIR

echo 'Staging files...'
rsync -a . $STAGEDIR --exclude=tmp --exclude=.git  --exclude=cache --exclude=__pycache__
echo 'Done staging.'

sudo chmod 777 -R $STAGEDIR

cd $STAGEDIR
echo 'Current dir: '`pwd`
# ------------------------------------------------

# Run job in a remote TPU VM


# VM_NAME=kmh-tpuvm-v3-32-5
# VM_NAME=kmh-tpuvm-v3-128-1
# VM_NAME=kmh-tpuvm-v3-8-2
# ZONE=europe-west4-a

# VM_NAME=kmh-tpuvm-v4-8-8
# ZONE=us-central2-b

VM_NAME=kmh-tpuvm-v3-32-preemptible-1
ZONE=europe-west4-a

echo $VM_NAME $ZONE

CONFIG=tpu

# some of the often modified hyperparametes:
batch=512
lr=0.0004
wd=0
ep=4000
warm=200
width=128
n_T=100
# n_T=18
drop=0.2

now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
TBNAME=${VM_NAME}_${CONFIG}_b${batch}_constlr${lr}_wd${wd}_adam_ep${ep}wm${warm}_w${width}_n${n_T}_dp${drop}_cond_fid_ncsnppedm_auglabel_0tcond_iddpm_b0.95_edmsampler_edmtrainer_edmweight_dbg1 # _endpre_iddpmcin
JOBNAME=hvae/iddpm/${now}_${salt}_${TBNAME}

LOGDIR=/kmh-nfs-ssd-eu-mount/logs/sqa/debug-km-code/$JOBNAME
sudo mkdir -p ${LOGDIR}
sudo chmod 777 -R ${LOGDIR}

# sudo chmod 777 /kmh-nfs-us-mount/data/cached  # for saving cached data

echo 'Log dir: '$LOGDIR
echo 'tb entry: '${TBNAME:10}:$LOGDIR  # remove the first 10 characters 'kmh-tpuvm-'

export CONDA_PATH=$(which conda)
export CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR
echo Current dir: $(pwd)

pip3 install tqdm

source $CONDA_INIT_SH_PATH
conda activate NNXeval

python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root=CIFAR \
    --config.model.image_size=32 \
    --config.model.out_channels=3 \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.lr_schedule=const \
    --config.weight_decay=${wd} \
    --config.optimizer=adamw \
    --config.adam_b2=0.95 \
    --config.warmup_epochs=${warm} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=100 \
    --config.visualize_per_epoch=100 \
    --config.eval_per_epoch=100 \
    --config.fid.fid_per_epoch=100 \
    --config.model.base_width=${width} \
    --config.model.n_T=${n_T} \
    --config.fid.eval_only=False \
    --config.aug.use_edm_aug=True \
    --config.model.use_aug_label=True \
    --config.model.dropout=${drop} \
    --config.model.net_type='ncsnppedm' \
    --config.model.t_conditioned=False \
    --config.model.M=1000 \
    --config.fid.device_batch_size=256 \
    --config.restore=/kmh-nfs-ssd-eu-mount/logs/kaiminghe/hvae/iddpm/20250101_023656_4ht1e4_kmh-tpuvm-v4-8-8_tpu_b512_constlr0.0004_wd0_adam_ep4000wm200_w128_n18_dp0.2_uncond_fid_ncsnppedm_auglabel_0tcond_iddpm_b0.95_edmsampler_edmtrainer_edmweight_dbg1 \
    --just_evaluate \
" 2>&1 | tee -a $LOGDIR/output.log

