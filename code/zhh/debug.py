import jax
import jax.numpy as jnp
from functools import wraps


_DEBUG = False

def set_debug():
    """
    Set the global debug flag to True. This will disable all JAX JIT compilation.

    Note: Please call it **as early as possible**. Using it in the `main` module may not work.
    """
    global _DEBUG
    _DEBUG = True
    print('[WARNING] Jit is disabled due to debug mode.')

def get_mode():
    """
    Get the global debug flag.
    """
    return _DEBUG

def _make_only_debug(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if hasattr(f,'_called'):
            return
        if not get_mode():
            f._called = True
            print(f'[WARNING] debug mode is off, function {f.__name__} will not work.')
            return
        return f(*args, **kwargs)
    return wrapper

@_make_only_debug
def print_stat(*args):
    """
    Use just like print, but for tensor, print their shape, dtype, max, min, mean, std.
    """
    last = False
    for arg in args:
        if 'tracer' in str(type(arg)).lower():
            print(arg.dtype, end=' ')
            jax.debug.print('{shape} max {max} min {min} mean {mean} std {std}',shape=arg.shape, max=jnp.max(arg), min=jnp.min(arg), mean=jnp.mean(arg), std=jnp.std(arg), ordered=True)  
            last = True
        elif hasattr(arg,'shape'):
            print(arg.dtype, arg.shape, 'max:', jnp.max(arg), 'min:', jnp.min(arg), 'mean:', jnp.mean(arg), 'std:', jnp.std(arg), end=' ')
        else:
            print(arg, end=' ')
    if not last: print()

@_make_only_debug
def print_tensor(*args, full=False):
    """
    All tensor values will be printed.

    Args:
        full (bool): If True, print the full tensor values.
    """
    for arg in args:
        if 'tracer' in str(type(arg)).lower():
            jax.debug.callback(lambda u: print(u.tolist(),end=' ') if full else print(u,end=' '),arg,ordered=True)
        elif hasattr(arg,'shape'):
            print(arg if not full else arg.tolist(),end=' ')
        else:
            print(arg, end=' ')
    print()


def printable_module(desc=''):
    """
    Debug util, useful in nn.Sequential.

    ### Example:

    >>> model = nn.Sequential([
    ...     nn.Dense(128),
    ...     printable_module('First'),
    ...     nn.relu,
    ...     nn.Dense(10),
    ... ])
    >>> x = jnp.zeros(7, 784)
    >>> model(x) # prints 'First (7, 128)'
    """
    return lambda x: print_stat(x, desc) or x