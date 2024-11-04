import jax
import zhh
import jax.numpy as jnp
from flax.linen import (
    Module as _Module,
    Dropout as _Dropout,
    Dense as _Dense,
    gelu as _gelu,
)
import flax.linen.initializers as initializers
import math
from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes
import optax
import jax.random as jr
from zhh.debug import get_mode
from functools import partial, wraps
import os
from typing import Callable
import json
import inspect

# @partial(jax.jit, static_argnames=('has_aux','update'))
def _update(state:TrainState, *args, has_aux=True, update=True, rng=None, add_training=False, **kwargs):
    if add_training:
        kwargs['training'] = update
    if rng is not None:
        kwargs['rng'] = rng
    if has_aux:
        def loss_fn(params):
            a =  state.apply_fn({'params': params}, *args, **kwargs)
            return (a[0],a[1:]) # with jax specification, can at most return two values
    else:
        def loss_fn(params):
            return state.apply_fn({'params': params}, *args, **kwargs)
    if update:
        results, grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(state.params)
        return results, state.apply_gradients(grads=grads)
    else:
        results = loss_fn(state.params)
        return results, state

_jit_update = partial(jax.jit, static_argnames=('has_aux','update', 'add_training'))(_update)

class ModuleWrapper:
    """
    This class is a wrapper designed to simplify the Jax training process by providing a PyTorch-like interface.

    ### Args:
    - `model`: a flax Module, must return the loss as the first element in the `__call__` method.
    - `optimizer`: an optax optimizer.
    - `has_aux`: whether the `__call__` method returns an auxiliary value (i.e. other than `loss`), default `True`.

    One caveat: due to the internal implementation of Jax, you can't know the model parameters **before** you called `step` for the first time. 

    ### Example:

    >>> import flax.linen as nn
    >>> class MLP(nn.Module):
    ...         @nn.compact
    ... def forward(self, x):
    ...     x = x.reshape((x.shape[0], -1))
    ...     x = nn.Dense(features=128)(x)
    ...     x = nn.relu(x)
    ...     x = nn.Dense(features=10)(x)
    ...     return x
    ... 
    ... def __call__(self, x, y):
    ...     logits = self.forward(x)
    ...     acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    ...     return F.cross_entropy(logits, y), acc
    >>> model = ModuleWrapper(MLP(), optimizer=optax.adam(1e-3))
    """
    def __init__(self, model: _Module, optimizer, has_aux=True):
        self._model = model
        assert hasattr(model, '__call__'), 'Your model don\'t have the `__call__` method.'
        # if not hasattr(model, '__call__'):
            # model.__call__ = model.get_loss
        self._optimizer = optimizer
        self.has_aux = has_aux
        sig = inspect.signature(self._model.__call__)
        self._model_has_train_arg = 'training' in sig.parameters or 'train' in sig.parameters
        self._model_train_arg = 'training' if 'training' in sig.parameters else 'train'
        self._model_is_random = 'rng' in sig.parameters
        self._rng = jax.random.PRNGKey(0)
        self._state = None
        self._is_loaded = False

    def set_input_shape(self, *args, **kwargs):
        raise DeprecationWarning('The method `set_input_shape` is deprecated!')

    def _set_input_shape(self, *args, **kwargs):
        # check that there are no methods overriding the model
        for method in ['step','save','load']:
            if hasattr(self._model, method):
                raise RuntimeError(f'Your model has a method named `{method}`, which is not allowed. Use another name.')
        if self._model_has_train_arg:
            kwargs[self._model_train_arg] = False
        if self._model_is_random:
            self._rng, use = jr.split(self._rng)
            kwargs['rng'] = use
        self._rng, use = jr.split(self._rng)
        self._params = self._model.init(use, *args, **kwargs)['params']
        self._get_loss_fn = lambda *params_and_args, **new_kwargs: self._model.apply(*params_and_args, **new_kwargs)
        self._state = TrainState.create(
            apply_fn=self._get_loss_fn,
            params=self._params,
            tx=self._optimizer
        )
        return self
    
    def step(self, *args,  update=True, **kwargs):
        """
        Update the wrapped model with one optimizer step.

        Args:
            `args`: the arguments of your `__call__` method
            `update`: whether to update the model, default `True`, use `False` for evaluation
            `kwargs`: the keyword arguments of your `__call__` method

        Returns:
            `results`: the return values of your `__call__` method. Meanwhile, the model is also updated.
        """
        if self._state is None:
            self._set_input_shape(*args, **kwargs) # lazy mode
        meth = _jit_update if not get_mode() else _update
        self._rng, use = jr.split(self._rng)
        results, next_state = meth(self._state, *args, has_aux=self.has_aux, update=update, add_training= self._model_has_train_arg, rng=use if self._model_is_random else None, **kwargs)
        self._state = next_state
        if self.has_aux:
            results = (results[0], *results[1]) # re-combine to the original form
        return results

    def reset_optimizer(self, optimizer):
        """
        Reset the optimizer of the model.

        ### Example:

        >>> model = ModuleWrapper(MLP(), optimizer=optax.adam(1e-3))
        >>> model.step(x, y)
        >>> model.reset_optimizer(optax.adam(1e-4))
        >>> model.step(x, y) # This update will use the new optimizer
        """
        self._optimizer = optimizer
        if self._state is not None:
            self._state = self._state.replace(tx=optimizer)
        return self

    def save(self, path):
        """
        Save the model and optimizer status to a file.

        Args:w
            `path`: the path to save, usually ends with `.pkl`
        """
        father_dir = os.path.dirname(path)
        if father_dir:
            os.makedirs(father_dir, exist_ok=True) # This solves the pytorch problem
        with open(path, 'wb') as f:
            f.write(to_bytes(self._state))

    def load(self, path):
        """
        Load the model and optimizer status from a file.

        Args:
            `path`: the path to load from
        """
        self._is_loaded = True
        with open(path, 'rb') as f:
            dic = from_bytes(self._state, f.read())
        self._get_loss_fn = lambda *params_and_args, **new_kwargs: self._model.apply(*params_and_args, **new_kwargs)
        self._state = TrainState.create(
            apply_fn=self._get_loss_fn,
            params=dic['params'],
            tx=self._optimizer
        )
        return self

    def __getattr__(self, name):
        if self._state is None and not self._is_loaded:
            raise RuntimeError(f'You called a method before calling model.step. This is unsupported.')
        if name in dir(ModuleWrapper):
            return lambda *args, **kwargs: getattr(ModuleWrapper, name)(self, *args, **kwargs)
        if name in self.__dict__:
            return self.__dict__[name]
        if hasattr(self._model, name):
            attr = getattr(self._model, name)
            if callable(attr):
                # use jit to speed up
                fast_attr = lambda *args, **kwargs: self._model.apply({'params': self._state.params}, *args, **kwargs, method=attr.__name__)
                return jax.jit(fast_attr) if not get_mode() else fast_attr
        raise AttributeError(f'\'{self.__class__.__name__}\' object has no attribute \'{name}\'')

    def num_parameters(self):
        """
        Count the number of parameters in the model. Can only be called **after** `step`.

        Returns:
            `int`: the number of parameters
        """
        return sum(p.size for p in jax.tree_flatten(self._state.params)[0])

    def __str__(self):
        """
        Print the model structure. Can only be called **after** `step`.
        """
        dic = jax.tree_map(lambda x:f'Tensor{x.shape}' if isinstance(x, jnp.ndarray) else x, self._state.params)
        return json.dumps(dic, indent=4)

    
    __repr__ = __str__

torch_weight_initializer = initializers.variance_scaling(scale=1/math.sqrt(3), mode='fan_in', distribution='uniform')
def torch_bias_initializer(in_size):
    scale = 1/math.sqrt(in_size)
    return lambda key, shape, dtype: jax.random.uniform(key, shape, dtype) * 2 * scale - scale

class TorchLinear(_Module):
    """
    The flax.linen `Dense` Module has a very bad default initialization. We changed it to torch-like initialization.

    In this module, if not specified, the weight and bias will be initialized just as https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
    """
    in_features: int
    out_features: int
    use_bias: bool = True
    dtype = None
    param_dtype = jnp.float32
    precision = None
    kernel_init = None
    bias_init = None
    dot_general = None
    dot_general_cls = None

    def setup(self):
        self._model = _Dense(
            features=self.out_features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=torch_weight_initializer,
            bias_init=torch_bias_initializer(self.in_features) if self.bias_init is None else self.bias_init,
            dot_general=self.dot_general,
            dot_general_cls=self.dot_general_cls
        )
    
    def __call__(self, x):
        return self._model(x)

class EmbedLinear(_Module):
    in_features: int
    out_features: int
    use_bias: bool = True
    dtype = None
    param_dtype = jnp.float32
    precision = None
    kernel_init = None
    bias_init = None
    dot_general = None
    dot_general_cls = None

    def setup(self):
        assert self.bias_init is None, 'The bias_init is not supported in EmbedLinear'
        self._model = _Dense(
            features=self.out_features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=torch_bias_initializer(self.in_features),
            bias_init=torch_bias_initializer(self.in_features),
            dot_general=self.dot_general,
            dot_general_cls=self.dot_general_cls
        )
    
    def __call__(self, x):
        return self._model(x)

# activations

GELU = partial(_gelu, approximate=False)
