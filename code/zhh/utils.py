"""
This is a collection of utility functions.
"""
import math, os
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import time

class Avger(list):
    """
    A automatically average calculator, very handy for loss updates.

    ### Example:

    >>> losses, acces = Avger(), Avger()
    >>> for i,(x,y) in enumerate(train_dl):
    ...     loss, acc = model.step(x, y)
    ...     losses.append(loss)
    ...     acces.append(acc)
    ...     if i % 100 == 0:
    ...         print(f'Loss: {losses}, Acc: {acces}')
    """
    def __str__(self):
        return f'{sum(self) / len(self):.4f}' if len(self) > 0 else 'N/A'
    
class Timer:
    """
    A simple timer class.

    Args:
        desc (str): Description of the timer
        print_fn (callable): A function to print the result

    ### Example:

    >>> with Timer('Epoch 0'):
    ...     for i in range(100):
    ...         time.sleep(1)
    >>> # Epoch 0 took 1:40 minutes
    """
    def __init__(self, desc='', print_fn=print):
        self.desc = desc
        self.print_fn = print_fn

    def __enter__(self):
        self.start = time.time()
        self.print_fn(f'{self.desc} starts ...')
        return self

    def __exit__(self, *args):
        self.end = time.time()
        # display time in hours, minutes and seconds
        dur = self.end - self.start
        h = int(dur // 3600)
        m = int((dur % 3600) // 60)
        s = int(dur % 60)
        time_desc = ''
        if h > 0:
            time_desc = f'{h}:{m:02d}:{s:02d} hours'
        elif m > 0:
            time_desc = f'{m}:{s:02d} minutes'
        else:
            time_desc = f'{s} seconds'
        self.print_fn(f'{self.desc} took {time_desc}')

def save_one_img(img, fp):
    """
    Save a single image into a file. Image should be (1, H, W, C) or (H, W, C)
    """

    if isinstance(fp,str):
        fat = os.path.dirname(fp)
        if fat: 
            os.makedirs(fat, exist_ok=True)
    if img.ndim == 4:
        img = img[0]
    img = jax.device_get(img)
    img = (np.array(img) * 255).astype(np.uint8)
    im = Image.fromarray(img.copy())
    im.save(fp)


def save_image_grid(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format_img=None):
    """Make a grid of images and Save it into an image file.

    (The code is copied from [here](https://github.com/google/flax/blob/main/examples/vae/utils.py))
    
    Args:
        ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
        fp:  A filename(string) or file object
        nrow (int, optional): Number of images displayed in each row of the grid.
        The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
        format_img(Optional):  If omitted, the format to use is determined from the
        filename extension. If a file object was used instead of a filename,
        this parameter should always be used.
    """
    if isinstance(fp,str):
        fat = os.path.dirname(fp)
        if fat: 
            os.makedirs(fat, exist_ok=True)
    if not (
        isinstance(ndarray, jnp.ndarray)
        or (
            isinstance(ndarray, list)
            and all(isinstance(t, jnp.ndarray) for t in ndarray)
        )
    ):
        raise TypeError(f'array_like of tensors expected, got {type(ndarray)}')

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = (
        int(ndarray.shape[1] + padding),
        int(ndarray.shape[2] + padding),
    )
    num_channels = ndarray.shape[3]
    grid = jnp.full(
        (height * ymaps + padding, width * xmaps + padding, num_channels),
        pad_value,
    ).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.at[
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ].set(ndarray[k])
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = np.array(jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8))
    im = Image.fromarray(ndarr.copy())
    im.save(fp, format=format_img)