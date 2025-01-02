source ka.sh # import VM_NAME, ZONE

echo 'solve'

# gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
# --worker=1 --command "
# ps -ef | grep -i unattended | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$2}'
# ps -ef | grep -i unattended | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$2}' | sh
# ps -ef | grep -i unattended | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$2}' | sh
# sleep 5
# sudo apt-get -y update
# sudo apt-get -y install nfs-common
# ps -ef | grep -i unattended | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$2}'
# ps -ef | grep -i unattended | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$2}' | sh
# ps -ef | grep -i unattended | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$2}' | sh
# sleep 6
# "

# for i in {1..10}; do echo Mount Mount 妈妈; done
# sleep 7

# gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
# --worker=1 --command "
# sudo mkdir -p /kmh-nfs-us-mount
# sudo mount -o vers=3 10.26.72.146:/kmh_nfs_us /kmh-nfs-us-mount
# sudo chmod go+rw /kmh-nfs-us-mount
# ls /kmh-nfs-us-mount

# sudo mkdir -p /kmh-nfs-ssd-eu-mount
# sudo mount -o vers=3 10.150.179.250:/kmh_nfs_ssd_eu /kmh-nfs-ssd-eu-mount
# sudo chmod go+rw /kmh-nfs-ssd-eu-mount
# ls /kmh-nfs-ssd-eu-mount
# "

# return 0

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all \
    --command "
sudo lsof -w /dev/accel0 | grep python | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$2}'
sudo lsof -w /dev/accel0 | grep python | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$2}' | sh
" # &> /dev/null
echo 'solved!'


# if test remote env partial workers
#
# sudo lsof -w /dev/accel0 | grep python | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$2}'
# sudo lsof -w /dev/accel0 | grep python | grep -v 'grep' | awk '{print \"sudo kill -9 \" \$2}' | sh


# pip install orbax-checkpoint==0.6.4 optree==0.13.0
