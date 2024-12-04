# Flow Matching

This code is copied from Kaiming's rectified flow, but we want to change it into nnx for further use.

This `FM` branch serves as the main branch of the repo. Any other modifications are based on this branch, and will merge useful stuff into this branch.

## FID mu and sigma

For CIFAR10:

If use pytorch dataloader (EDM):  
`/kmh-nfs-us-mount/data/cached/cifar10_jax_stats_20240820.npz`  
If use tfds (with random flip):  
`/kmh-nfs-us-mount/staging/zhh/data/cached/zhh_tfds_train_cifar10_stats_20241124.npz`

## t network

After testing, `sqa_t_ver1` reaches the best performance with the least number of parameters. Sigmoid should not be used, and relu activation is the best (lrelu is equally good). The checkpoint is at `/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20241128_031750_8xab8k_kmh-tpuvm-v2-32-preemptible-2__b_lr_ep_eval/checkpoint_4850`

## checkpoints

Vanilla FM baseline: `/kmh-nfs-us-mount/logs/sqa/sqa_Flow_matching/20241127_010129_q9ifcu_kmh-tpuvm-v2-32-6__b_lr_ep_eval/checkpoint_194000`

t_network: `/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20241128_031750_8xab8k_kmh-tpuvm-v2-32-preemptible-2__b_lr_ep_eval/checkpoint_4850`

FM with no t baseline: `/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20241130_140252_c95a47_kmh-tpuvm-v3-32-preemptible-1__b_lr_ep_eval/checkpoint_194000`

DDIM: `/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20241202_214940_zd8s4j_kmh-tpuvm-v3-32-1__b_lr_ep_eval/checkpoint_48500`

## datasets

We now use pytorch dataset. To use tfds, ???