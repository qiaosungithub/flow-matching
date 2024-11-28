# Flow Matching

This code is copied from Kaiming's rectified flow, but we want to change it into nnx for further use.

## FID mu and sigma

For CIFAR10:

If use pytorch dataloader (EDM):  
`/kmh-nfs-us-mount/data/cached/cifar10_jax_stats_20240820.npz`  
If use tfds (with random flip):  
`/kmh-nfs-us-mount/staging/zhh/data/cached/zhh_tfds_train_cifar10_stats_20241124.npz`

## t network

After testing, `sqa_t_ver1` reaches the best performance with the least number of parameters. Sigmoid should not be used, and relu activation is the best (lrelu is equally good). The checkpoint is at `/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20241128_031750_8xab8k_kmh-tpuvm-v2-32-preemptible-2__b_lr_ep_eval/checkpoint_4850`