# 卡.sh
source config.sh

if [ -z "$OWN_CONDA_ENV_NAME" ]; then
    echo "Please set your own config.sh. See README for reference"
    sleep 60
    exit 1
fi

if [ -z "$TASKNAME" ]; then
    echo "Please set your own config.sh. See README for reference"
    sleep 60
    exit 1
fi

# specify your TPU VM name here!

############## TPU VMs ##############

# export VM_NAME=kmh-tpuvm-v2-32-1
export VM_NAME=kmh-tpuvm-v2-32-2
# export VM_NAME=kmh-tpuvm-v2-32-3
# export VM_NAME=kmh-tpuvm-v3-32-1

#####################################

# Zone: your TPU VM zone

if [[ $VM_NAME == *"v3"* ]]; then
    export ZONE=europe-west4-a
else
    export ZONE=us-central1-a
fi

# DATA_ROOT: the disk mounted
# FAKE_DATA_ROOT: the fake data (imagenet_fake) link
# USE_CONDA: 1 for europe, 2 for us (common conda env)

if [[ $ZONE == *"europe"* ]]; then
    export DATA_ROOT="kmh-nfs-ssd-eu-mount"
    export FAKE_DATA_ROOT=$EU_IMAGENET_FAKE
    export USE_CONDA=1
else
    export DATA_ROOT="kmh-nfs-us-mount"
    export USE_CONDA=2
    export FAKE_DATA_ROOT=$US_IMAGENET_FAKE
fi