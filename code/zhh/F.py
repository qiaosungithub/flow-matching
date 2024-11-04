"""
This module is designed to simulate the interface of PyTorch's `F` module, at least in common usages.
"""
import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as jr

def cross_entropy(preds, targets):
    """
    This resembles PyTorch: `targets` can be a 1D tensor of shape (B,) (class indices) or a 2D tensor of shape (B, C) (probabilities)
    """
    if len(targets.shape) == 1:
        targets = nn.one_hot(targets, preds.shape[-1])
    return - jnp.mean(jax.nn.log_softmax(preds) * targets)

def mse_loss(preds, targets):
    return jnp.mean((preds - targets)**2)

def KL(mu1, logvar1, mu2=0, logvar2=1):
    """
    KL(N(mu1, exp(logvar1)) || N(mu2, exp(logvar2))

    Note that the KL is calculated element-wise, and then averaged.
    """
    return 0.5 * jnp.mean(logvar2 - logvar1 + (jnp.exp(logvar1) + (mu1 - mu2)**2) / jnp.exp(logvar2) - 1)

def patchify(img, patch_size):
    """
    This function mimic the behavior of `torch.nn.functional.unfold` in PyTorch.

    Args:
        img: (B, H, W, C) tensor
        patch_size: int

    Returns:
        (B, `num_patches`, `patch_size` \* `patch_size` \* C) tensor

    Example:
    >>> img = jnp.arange(48).reshape(1, 4, 4, 3)
    >>> img
    >>> [[[[ 0  1  2]
    ...     [ 3  4  5]
    ...     [ 6  7  8]
    ...     [ 9 10 11]]
    ...  
    ...     [[12 13 14]
    ...     [15 16 17]
    ...     [18 19 20]
    ...     [21 22 23]]
    ...  
    ...     [[24 25 26]
    ...     [27 28 29]
    ...     [30 31 32]
    ...     [33 34 35]]
    ...  
    ...     [[36 37 38]
    ...     [39 40 41]
    ...     [42 43 44]
    ...     [45 46 47]]]]
    >>> patchify(img, 2)
    >>>   [[[ 0  1  2  3  4  5 12 13 14 15 16 17]
    ...     [ 6  7  8  9 10 11 18 19 20 21 22 23]
    ...     [24 25 26 27 28 29 36 37 38 39 40 41]
    ...     [30 31 32 33 34 35 42 43 44 45 46 47]]]
    """
    B, H, W, C = img.shape
    p = patch_size
    assert H % p == 0 and W % p == 0, 'Image dimensions must be divisible by the patch size.'
    img = img.reshape(B, H//p, p, W//p, p, C)
    return img.transpose(0, 1, 3, 2, 4, 5).reshape(B, -1, p*p*C)

def dropout(x, rate, training, rng):
    """
    This function mimic the behavior of `torch.nn.functional.dropout` in PyTorch.

    Args:
        x: input tensor
        rate: dropout rate
        training: bool, whether to apply dropout (required!)
        rng: random key (required!)

    Returns:
        tensor
    """
    assert isinstance(training, bool), "Get an invalid `training` argument. Did you forget to pass in `training` in the forward pass?"
    if not training:
        return x
    mask = (jr.uniform(rng, x.shape) > rate).astype(jnp.float32)
    return x * mask / (1 - rate)

def stochastic_depth(input, p, training, rng, mode='batch'):
    """
    Stochastic Depth from "Deep Networks with Stochastic Depth" (https://arxiv.org/abs/1603.09382)
    
    Copied from https://pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        rng: random number generator
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor [N, ...]: The randomly zeroed tensor.
    """
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    elif mode == "batch":
        size = [1] * input.ndim
    else:
        raise ValueError("Unknown mode: {}".format(mode))
    noise = (jr.uniform(rng, size) < survival_rate).astype(jnp.float32)
    if survival_rate > 0.0:
        noise/=survival_rate
    return input * noise

if __name__ == '__main__':
    img = jnp.arange(48).reshape(1, 4, 4, 3)
    print(img)
    print(patchify(img, 2)) # (1, 4, 12)