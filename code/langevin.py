import jax
import os
# from math import sqrt
import jax.numpy as jnp
from utils.utils import save_img
from jax import random
from jax import lax

# TODO

def apply_langevin(state, x, alpha, noise, indices, mask=None):
    grad, _, _ = state.apply_fn(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state, False, x, indices)
    if mask is not None:
        x = x + (alpha / 2 * grad + jnp.sqrt(alpha) * noise) * (1-mask)
    else:
        x = x + alpha / 2 * grad + jnp.sqrt(alpha) * noise
    return x, grad

fast_apply_langevin = jax.pmap(
    apply_langevin,
    axis_name='batch',
)

def langevin(state, shape, sigmas, eps, T, rngs, whole_process=False, clamp=False, verbose=False, show_freq=1):
    """
    rngs: a Rng class instance
    """

    assert len(shape) == 4
    # it's better not to clamp
    bs = shape[0]
    x = jax.random.normal(rngs.evaluation(), shape=shape) # we only need 1 tpu to calculate
    if whole_process:
        assert bs <= 20, "batch size should be less than 20 if you want to save the whole process"
        all_samples = []
    
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        alpha = eps * (sigma ** 2) / (sigmas[-1] ** 2)
        indices = i * jnp.ones(bs, dtype=jnp.int32)
        assert indices.shape == (bs,)
        for t in range(T):
            noise = jax.random.normal(rngs.evaluation(), shape=x.shape)
            # print("indices: ", indices)
            # print("indices.shape", indices.shape)
            # replicate the variables to all devices
            num_replicas = jax.local_device_count()
            xr = jnp.tile(x.reshape(1, *x.shape), (num_replicas, 1, 1, 1, 1))
            alphar = alpha * jnp.ones(num_replicas)
            noiser = jnp.tile(noise.reshape(1, *noise.shape), (num_replicas, 1, 1, 1, 1))
            indicesr = jnp.tile(indices.reshape(1, *indices.shape), (num_replicas, 1))

            # print("xr.shape", xr.shape)
            # print("alphar.shape", alphar.shape)
            # print("noiser.shape", noiser.shape)
            # print("indicesr.shape", indicesr.shape)

            x, grad = fast_apply_langevin(state, xr, alphar, noiser, indicesr)
            # print("x.shape", x.shape)
            # print("grad.shape", grad.shape)
            x = x[0]
            grad = grad[0]
            if clamp:
                x = jnp.clip(x, 0, 1)
            if verbose:
                grad_norm = jnp.linalg.norm(grad.reshape(bs, -1), axis=1).mean()
                image_norm = jnp.linalg.norm(x.reshape(bs, -1), axis=1).mean()
                noise_norm = jnp.linalg.norm(noise.reshape(noise.shape[0], -1), axis=-1).mean()
                snr = jnp.sqrt(alpha) * grad_norm / noise_norm # signal to noise ratio
                grad_mean_norm = jnp.linalg.norm(grad.mean(axis=0).reshape(-1)) ** 2 * sigma ** 2
                if jax.process_index() == 0:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                                    i, alpha, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()), flush=True)
        if whole_process and i % show_freq == 0:
            all_samples.append(x)

    if whole_process:
        all_samples = jnp.stack(all_samples, axis=0)
        all_samples = all_samples.reshape(-1, *shape[1:])
        return x, all_samples
    else:
        return x

def langevin_masked(state, x, sigmas, eps, T, rngs, mask, whole_process=False, clamp=False, verbose=False):
    """
    INPUT---
    rngs: a Rng class instance
    x: with shape (num_replicas, bs, h, w, c)
    mask: with shape (num_replicas, bs, h, w, c)
    OUTPUT---
    x: with shape (num_replicas, bs, h, w, c)
    all_samples: with shape (bs * n_noise_levels, h, w, c)
    note: if we want to save the whole process, x is the same for each replica, with local_bs = 10; otherwise we have 64 samples
    """

    # it's better not to clamp
    # print("--------------------in langevin_masked----------------------")
    num_replicas = x.shape[0]
    local_bs = x.shape[1]
    bs = num_replicas * local_bs
    assert mask.shape == x.shape
    if whole_process:
        assert local_bs <= 20, "batch size should be less than 20 if you want to save the whole process"
        all_samples = []
    
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        alpha = eps * (sigma ** 2) / (sigmas[-1] ** 2)
        indices = i * jnp.ones((num_replicas, local_bs), dtype=jnp.int32)
        for t in range(T):
            noise = jax.random.normal(rngs.evaluation(), shape=x.shape)
            # replicate the variables to all devices
            alphar = alpha * jnp.ones(num_replicas)

            x, grad = fast_apply_langevin(state, x, alphar, noise, indices, mask=mask)

            if clamp:
                x = jnp.clip(x, 0, 1)
            if verbose:
                grad_norm = jnp.linalg.norm(grad.reshape(bs, -1), axis=1).mean()
                image_norm = jnp.linalg.norm(x.reshape(bs, -1), axis=1).mean()
                noise_norm = jnp.linalg.norm(noise.reshape(noise.shape[0], -1), axis=-1).mean()
                snr = jnp.sqrt(alpha) * grad_norm / noise_norm # signal to noise ratio
                grad_mean_norm = jnp.linalg.norm(grad.mean(axis=0).reshape(-1)) ** 2 * sigma ** 2
                if jax.process_index() == 0:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                                    i, alpha, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()), flush=True)
        if whole_process:
            # note that all device generate the same image
            # print("x.shape", x.shape)
            # print("x[0].shape", x[0].shape)
            all_samples.append(x[0])

    if whole_process:
        all_samples = jnp.stack(all_samples, axis=0)
        all_samples = all_samples.reshape(-1, *x.shape[2:])
        # print("all_samples.shape", all_samples.shape)
        return x, all_samples
    else:
        return x