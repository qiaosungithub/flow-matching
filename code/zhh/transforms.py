"""
This is going to provide an interface like `torchvision.transforms` in PyTorch.
"""
import jax
import jax.numpy as jnp
from zhh.models import ModuleWrapper
import jax.random as jr
import flax.linen as nn
from functools import partial, wraps
from typing import Iterable, Optional
import inspect

class Transform(nn.Module):
    """
    A base class for all image transforms.

    You inherit this class and implement the `__call__` method.

    Args:
        data: the input data
        rng: the random key (optional, but must have one if you want to use random operations)
    """

    def __call__(self, data, rng):
        raise NotImplementedError

    def run(self, data):
        if not hasattr(self, '_jit_apply'):
            self._jit_apply = jax.jit(self.apply)
            # check whether self requires `rng` as an argument
            sig = inspect.signature(self.__call__)
            self._rng_required = 'rng' in sig.parameters
        if self._rng_required:
            return self._jit_apply({}, data, rng=self.key.getkey())
        return self._jit_apply({}, data)

class Compose(Transform):
    """
    Composes several transforms together.

    Args:
        *transforms: list/iterable of transforms to compose.
    """
    transforms: Iterable[Transform]

    # def __init__(self, *transforms):
    #     if isinstance(transforms[0], Transform):
    #         self.transforms = transforms
    #     else:
    #         self.transforms = transforms[0]

    def __call__(self, data):
        for t in self.transforms:
            data = t.run(data)
        return data

class ToTensor(Transform):
    """
    Convert an image into a tensor, ranging in [0, 1].
    """

    # no jax.jit, since data is not a jax array
    def __call__(self, data):
        return jnp.float32(data) / 255.
    
class Normalize(Transform):
    """
    Normalize an image: substract `mean` and divide by `std`.

    Args:
        mean: the mean value to substract, shape should be (`channels`,) 
        std: the standard deviation to divide by, shape should be (`channels`,)
        eps: a small value to avoid division by zero

    ### Example
    
    >>> import zhh.random as zr
    >>> foo = zr.rand(10, 4)
    >>> tr = Normalize(mean=0.5, std=0.58)
    >>> tr(foo).mean(), tr(foo).std() # -0.007937512 0.49972156
    """
    mean: jnp.ndarray
    std: jnp.ndarray
    eps: float = 1e-6

    def __call__(self, data):
        return (data - jnp.array(self.mean).reshape(1,1,1,-1)) / (jnp.array(self.std).reshape(1,1,1,-1) + self.eps)

class RandomHorizontalFlip(Transform):
    """
    Randomly flip an image horizontally. Image must be (B, H, W, C)

    Args:
        p: the probability of flipping
    """
    p: float = 0.5
    key = jr.PRNGKey(0)

    def __call__(self, data: jnp.ndarray, rng):
        assert data.ndim == 4, 'Image must be (B, H, W, C)'
        do_flip = jr.uniform(rng, (data.shape[0], 1, 1, 1)) < self.p
        return jnp.where(do_flip, jnp.flip(data, axis=-2), data)

class RandomVerticalFlip(Transform):
    """
    Randomly flip an image vertically. Image must be (B, H, W, C)

    Args:
        p: the probability of flipping
    """
    p: float = 0.5
    key = jr.PRNGKey(0)

    def __call__(self, data, rng):
        assert data.ndim == 4, 'Image must be (B, H, W, C)'
        do_flip = jr.uniform(rng, (data.shape[0], 1, 1, 1)) < self.p
        return jnp.where(do_flip, jnp.flip(data, axis=-3), data)
    
class RandomCrop(Transform):
    """
    Randomly crop an image. Image must be (B, H, W, C)

    Args:
        size: the size of the output image
        padding: how many pixels to pad the image before cropping

    ### Example

    >>> import zhh.random as zr
    >>> foo = zr.randn(2, 4, 4, 3)
    >>> tr = RandomCrop(3, padding=1)
    >>> tr(foo).shape # (2, 3, 3, 3)
    """
    size: Iterable[int] | int
    padding: Optional[int] = None
    key = jr.PRNGKey(0)

    @property
    def getsize(self):
        return self.size if isinstance(self.size, tuple) else (self.size, self.size)

    @staticmethod
    @partial(jax.vmap, in_axes=(0, 0, 0, None, None))
    def _crop_img(single_img, start_x, start_y, out_h, out_w):
        return jax.lax.dynamic_slice(single_img, (start_y, start_x, 0), (out_h, out_w, single_img.shape[-1]))

    def __call__(self, data, rng):
        assert data.ndim == 4, 'Image must be (B, H, W, C)'
        if self.padding:
            data = jnp.pad(data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        data_h, data_w = data.shape[1], data.shape[2]
        out_h, out_w = self.getsize
        start_x = jr.randint(rng, (data.shape[0],), 0, data_w - out_w)
        start_y = jr.randint(rng, (data.shape[0],), 0, data_h - out_h)
        data = self._crop_img(data, start_x, start_y, out_h, out_w)
        return data

class RandomRotation(Transform):
    """
    Randomly rotate an image (about its center). Image must be (B, H, W, C)

    This is torchvision's RandomRotation with "NEAREST" interpolation.

    Args:
        degrees: the **maximal** abs value of degrees to rotate
        fill: when rotating, fill the empty space with this value, default is 0.

    ### Example

    >>> import zhh.random as zr
    >>> foo = zr.randn(2, 4, 4, 3)
    >>> tr = RandomCrop(3, padding=1)
    >>> tr(foo).shape # (2, 3, 3, 3)
    """
    degrees: float
    fill: float = 0.
    key = jr.PRNGKey(0)

    def setup(self):
        print('Warning: our RandomRotation is currently not very optimized. Thus, the code may slow down, especially for the first epoch.')

    @staticmethod
    @partial(jax.vmap, in_axes=(None, None, 0))
    def rotate_indices(h, w, angle):
        positions = jnp.indices((h, w)).astype(jnp.float32) # (2, h, w)
        # by default, origin is at center
        offset = jnp.array([h - 1, w - 1]) / 2
        rot_mat = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)]
        ])
        new_positions = jnp.einsum('ij,jhw->ihw', rot_mat, positions - offset[:, None, None]) + offset[:, None, None]
        return new_positions.round().astype(jnp.int32)

    @staticmethod
    @partial(jax.vmap, in_axes=(0, 0, None))
    def get_pixels(b, xy, img):
        # img is [H, W, C]
        return jax.lax.dynamic_slice(img, (b, xy[0], xy[1], 0), (1, 1, 1, img.shape[-1])).squeeze()

    @staticmethod
    def out_range(xy, h, w):
        return jnp.logical_or(
            jnp.logical_or(xy[..., 0] < 0, xy[..., 0] >= h),
            jnp.logical_or(xy[..., 1] < 0, xy[..., 1] >= w)
        )

    def __call__(self, data, rng):
        B, H, W, C = data.shape
        angle = jr.uniform(rng, (B,)) * (2 * self.degrees) - self.degrees # (B,)
        angle = jnp.deg2rad(angle)
        new_positions = self.rotate_indices(H, W, angle) # (B, 2, H, W)
        new_positions = new_positions.transpose(0, 2, 3, 1).reshape(-1,2) # (B*H*W, 2)
        b_vec = jnp.arange(B).reshape(-1, 1, 1).repeat(H, axis=1).repeat(W, axis=2).reshape(-1)
        pixels = self.get_pixels(b_vec, new_positions, data)
        out_range = self.out_range(new_positions, H, W).reshape(-1,1)
        return jnp.where(
            out_range,
            self.fill,
            pixels
        ).reshape(B, H, W, C)

# if __name__ == '__main__':
#     foo = zr.rand(10, 4)
#     tr = Normalize(mean=0.5, std=0.58)
#     print(tr.run(foo).mean(), tr.run(foo).std()) # -0.007937512 0.49972156