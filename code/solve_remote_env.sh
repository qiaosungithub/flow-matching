source ka.sh # import VM_NAME, ZONE

echo 'solve'
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all \
    --command "
pip3 install tensorstore==0.1.67
" # &> /dev/null
echo 'solved!'
