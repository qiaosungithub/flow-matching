# initialize and set up remote TPU VM
source ka.sh # import VM_NAME, ZONE

timeout 20s gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
sudo reboot
" --ssh-flag="-o ConnectionAttempts=1" --ssh-flag="-o ConnectTimeout=5"

# sleep 5 minutes
echo "Restart done. Sleeping for 5 minutes..."
sleep 300

# while the command doesn't return 0, continue to wait
RETURN_CODE=1
while [ $RETURN_CODE -ne 0 ]; do
    timeout 20s gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
    ls
    "
    RETURN_CODE=$?
    sleep 60
done

echo "VM is ready!"

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "

sudo mkdir -p /kmh-nfs-us-mount
sudo mount -o vers=3 10.26.72.146:/kmh_nfs_us /kmh-nfs-us-mount
sudo chmod go+rw /kmh-nfs-us-mount
ls /kmh-nfs-us-mount

sudo mkdir -p /kmh-nfs-ssd-eu-mount
sudo mount -o vers=3 10.150.179.250:/kmh_nfs_ssd_eu /kmh-nfs-ssd-eu-mount
sudo chmod go+rw /kmh-nfs-ssd-eu-mount
ls /kmh-nfs-ssd-eu-mount

"


# sudo apt-get -y update
# sudo apt-get -y install nfs-common