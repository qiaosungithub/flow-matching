source ka.sh # import VM_NAME, ZONE

echo 'solve'
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all \
    --command "
pip list | grep jax
pip list | grep orbax
pip list | grep optree
pip list | grep dtypes
" # &> /dev/null
echo 'solved!'


# pip install orbax-checkpoint==0.6.4 optree==0.13.0
