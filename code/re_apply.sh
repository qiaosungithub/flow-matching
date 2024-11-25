source ka.sh

if [[ $VM_NAME == *"v3"* ]]; then
    export ACCEL_TP="v3-32"
else
    export ACCEL_TP="v2-32"
fi

if [[ $VM_NAME == *"preemptible"* ]]; then
    echo "Creating preemptible TPU VM $VM_NAME"
    yes | gcloud compute tpus tpu-vm delete $VM_NAME --zone=$ZONE --quiet
    gcloud compute tpus tpu-vm create $VM_NAME \
    --zone=$ZONE \
    --accelerator-type=$ACCEL_TP \
    --version=tpu-ubuntu2204-base \
    --preemptible
    if [ "$(gcloud compute tpus describe $VM_NAME --zone=$ZONE --format="value(state)")" == "READY" ]; then
        echo "Now, TPU VM $VM_NAME is good, ready to use"
        bash init_preemptible_remote.sh
        bash test_remote_env.sh
    else
        echo "TPU is still not ready, please check"
        return 1
    fi
fi