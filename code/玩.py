import numpy as np
file = '/kmh-nfs-ssd-eu-mount/data/cached/zhh/imagenet32_train_jax_stats_20241228.npz'
with np.load(file) as data:
    if "ref_mu" in data:
        ref_mu, ref_sigma = data["ref_mu"], data["ref_sigma"]
    elif "mu" in data:
        ref_mu, ref_sigma = data["mu"], data["sigma"]
    else:
        raise NotImplementedError

print(ref_mu.shape, ref_sigma.shape)