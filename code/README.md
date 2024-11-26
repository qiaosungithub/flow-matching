# Flow Matching

This code is copied from Kaiming's rectified flow, but we want to change it into nnx for further use.

## FID mu and sigma

For CIFAR10:

If use pytorch dataloader (EDM):  
`/kmh-nfs-us-mount/data/cached/cifar10_jax_stats_20240820.npz`  
If use tfds (with random flip):  
`/kmh-nfs-us-mount/staging/zhh/data/cached/zhh_tfds_train_cifar10_stats_20241124.npz`