source ka.sh # import VM_NAME, ZONE

echo 'solve'
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all \
    --command "
pip install orbax-checkpoint==0.6.4 ml-dtypes==0.5.0 tensorstore==0.1.67
" # &> /dev/null
echo 'solved!'

# pip3 install tensorstore==0.1.67

# pip3 install flax==0.10.2
# pip3 show flax