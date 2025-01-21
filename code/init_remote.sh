# initialize and set up remote TPU VM
source ka.sh # import VM_NAME, ZONE

echo $VM_NAME $ZONE

STAGEDIR=/$DATA_ROOT/staging/$(whoami)/debug/init
sudo mkdir -p $STAGEDIR
sudo chmod 777 $STAGEDIR
echo 'Staging files...'
sudo rsync -a requirements.txt $STAGEDIR
sudo rsync -a 装牛牛X.sh $STAGEDIR
echo 'staging dir: '$STAGEDIR
echo 'Done staging.'

LOGDIR=$STAGEDIR/log
sudo rm -rf $LOGDIR
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}
echo 'Log dir: '$LOGDIR


# mount NFS Filestore
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "

sudo apt-get -y update
sudo apt-get -y install nfs-common

sudo mkdir -p /kmh-nfs-us-mount
sudo mount -o vers=3 10.26.72.146:/kmh_nfs_us /kmh-nfs-us-mount
sudo chmod go+rw /kmh-nfs-us-mount
ls /kmh-nfs-us-mount

sudo mkdir -p /kmh-nfs-ssd-eu-mount
sudo mount -o vers=3 10.150.179.250:/kmh_nfs_ssd_eu /kmh-nfs-ssd-eu-mount
sudo chmod go+rw /kmh-nfs-ssd-eu-mount
ls /kmh-nfs-ssd-eu-mount

"

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
sudo rm -rf /home/\$(whoami)/.local
cd $STAGEDIR
echo 'Current dir: '
pwd
source 装牛牛X.sh
"
