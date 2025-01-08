source ka.sh # import VM_NAME, ZONE

echo 'solve'
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all \
    --command "
ls /kmh-nfs-us-mount/code
if [ ! -d /kmh-nfs-us-mount/code ]; then
  echo '/kmh-nfs-us-mount does not exist' >&2
  exit 1
fi
" # &> /dev/null
echo 'solved!'

# pip3 install tensorstore==0.1.67