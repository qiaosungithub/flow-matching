source ka.sh # import VM_NAME, ZONE

echo 'solve'
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all \
    --command "
pip3 install orbax-checkpoint==0.6.4
" # &> /dev/null
echo 'solved!'

# pip3 install tensorstore==0.1.67
