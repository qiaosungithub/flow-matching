# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from typing import Any, Optional, Tuple
from . import layers
from . import up_or_down_sampling
# import flax.linen as nn
import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np

conv1x1 = layers.ddpm_conv1x1 # not used
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size = 256, scale = 1.0, rngs=None):
        self.embedding_size = embedding_size
        self.scale = scale
        self.rngs = rngs

        assert embedding_size % 2 == 0

        self.freqs = nn.Embed(num_embeddings=1, features=self.embedding_size // 2, embedding_init=jax.nn.initializers.normal(stddev=self.scale), rngs=rngs)

    def __call__(self, x):
        freqs = jax.lax.stop_gradient(self.freqs(jnp.zeros(1, dtype=jnp.int32)))
        freqs = jnp.squeeze(freqs, axis=0)
        assert x.ndim == 1
        x_proj = x[:, None] * freqs[None, :] * 2 * jnp.pi

        # this should be sin first; don't need to swap it outside
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

# class GaussianFourierProjection(nn.Module):
#     """Gaussian Fourier embeddings for noise levels."""

#     embedding_size: int = 256
#     scale: float = 1.0
#     freqs_name: str = 'freqs'  # was "W" in CM

#     @nn.compact
#     def __call__(self, x):
#         freqs = self.param(
#             self.freqs_name, jax.nn.initializers.normal(stddev=self.scale), (self.embedding_size,)
#         )
#         freqs = jax.lax.stop_gradient(freqs)
#         assert x.ndim == 1
#         x_proj = x[:, None] * freqs[None, :] * 2 * jnp.pi

#         # this should be sin first; don't need to swap it outside
#         return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

# class Combine(nn.Module):
#     """Combine information from skip connections."""

#     method: str = "cat"

#     @nn.compact
#     def __call__(self, x, y):
#         h = conv1x1(x, y.shape[-1])
#         if self.method == "cat":
#             return jnp.concatenate([h, y], axis=-1)
#         elif self.method == "sum":
#             return h + y
#         else:
#             raise ValueError(f"Method {self.method} not recognized.")

class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, in_dim, out_dim, method = "cat", rngs=None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.method = method
        self.rngs = rngs

        self.conv1x1 = conv1x1(in_dim, out_dim, rngs=rngs)

    def __call__(self, x, y):
        assert x.shape[-1] == self.in_dim
        assert y.shape[-1] == self.out_dim
        h = self.conv1x1(x)
        if self.method == "cat":
            return jnp.concatenate([h, y], axis=-1)
        elif self.method == "sum":
            return h + y
        else:
            raise ValueError(f"Method {self.method} not recognized.")


# class AttnBlockpp(nn.Module):
#     """Channel-wise self-attention block. Modified from DDPM."""

#     skip_rescale: bool = False
#     init_scale: float = 0.0

#     @nn.compact
#     def __call__(self, x):
#         B, H, W, C = x.shape
#         h = nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x)
#         q = NIN(C)(h)
#         k = NIN(C)(h)
#         v = NIN(C)(h)

#         w = jnp.einsum("bhwc,bHWc->bhwHW", q, k) * (int(C) ** (-0.5))
#         w = jnp.reshape(w, (B, H, W, H * W))
#         w = jax.nn.softmax(w, axis=-1)
#         w = jnp.reshape(w, (B, H, W, H, W))
#         h = jnp.einsum("bhwHW,bHWc->bhwc", w, v)
#         h = NIN(C, init_scale=self.init_scale)(h)
#         if not self.skip_rescale:
#             return x + h
#         else:
#             return (x + h) / np.sqrt(2.0)

class AttnBlockpp(nn.Module):
    """
    Channel-wise self-attention block. Modified from DDPM.

    in_channel = out_channel
    """

    def __init__(self, 
        in_planes,
        skip_rescale = False, 
        init_scale = 0.0, 
        rngs=None
    ):
        self.in_planes = in_planes
        self.skip_rescale = skip_rescale
        self.init_scale = init_scale
        self.rngs = rngs

        self.group_norm = nn.GroupNorm(num_features=in_planes, num_groups=min(self.in_planes // 4, 32), rngs=self.rngs)
        self.q_NIN = NIN(self.in_planes, rngs=self.rngs)
        self.k_NIN = NIN(self.in_planes, rngs=self.rngs)
        self.v_NIN = NIN(self.in_planes, rngs=self.rngs)

        self.final_NIN = NIN(self.in_planes, init_scale=init_scale, rngs=self.rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        assert C == self.in_planes
        h = self.group_norm(x)
        q = self.q_NIN(h)
        k = self.k_NIN(h)
        v = self.v_NIN(h)

        w = jnp.einsum("bhwc,bHWc->bhwHW", q, k) * (int(C) ** (-0.5))
        w = jnp.reshape(w, (B, H, W, H * W))
        w = jax.nn.softmax(w, axis=-1)
        w = jnp.reshape(w, (B, H, W, H, W))
        h = jnp.einsum("bhwHW,bHWc->bhwc", w, v)
        h = self.final_NIN(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


# class Upsample(nn.Module):
#     out_ch: Optional[int] = None
#     with_conv: bool = False
#     fir: bool = False
#     fir_kernel: Tuple[int] = (1, 3, 3, 1)

#     @nn.compact
#     def __call__(self, x):
#         B, H, W, C = x.shape
#         out_ch = self.out_ch if self.out_ch else C
#         if not self.fir:
#             h = jax.image.resize(x, (x.shape[0], H * 2, W * 2, C), "nearest")
#             if self.with_conv:
#                 h = conv3x3(h, out_ch)
#         else:
#             if not self.with_conv:
#                 h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
#             else:
#                 h = up_or_down_sampling.Conv2d(
#                     out_ch,
#                     kernel=3,
#                     up=True,
#                     resample_kernel=self.fir_kernel,
#                     use_bias=True,
#                     kernel_init=default_init(),
#                 )(x)

#         assert h.shape == (B, 2 * H, 2 * W, out_ch)
#         return h

class Upsample(nn.Module):

    def __init__(self, 
        in_planes,
        out_ch=None,
        with_conv=False, 
        fir=False, 
        fir_kernel=(1, 3, 3, 1), 
        rngs=None
    ):
        self.in_planes = in_planes
        self.out_ch = in_planes if out_ch is None else out_ch
        self.with_conv = with_conv
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.rngs = rngs

        if not self.fir:
            self.conv3x3 = conv3x3(in_planes, self.out_ch, rngs=self.rngs)
        elif self.with_conv:
            self.conv3x3 = up_or_down_sampling.Conv2d(
                in_dim=in_planes,
                fmaps=self.out_ch,
                kernel=3,
                up=True,
                resample_kernel=self.fir_kernel,
                use_bias=True,
                kernel_init=default_init(),
                rngs=self.rngs
            )

    def __call__(self, x):
        B, H, W, C = x.shape
        assert self.in_planes == C
        if not self.fir:
            h = jax.image.resize(x, (x.shape[0], H * 2, W * 2, C), "nearest")
            if self.with_conv:
                h = self.conv3x3(h)
        else:
            if not self.with_conv:
                h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = self.conv3x3(x)

        assert h.shape == (B, 2 * H, 2 * W, self.out_ch)
        return h


# class Downsample(nn.Module):
#     out_ch: Optional[int] = None
#     with_conv: bool = False
#     fir: bool = False
#     fir_kernel: Tuple[int] = (1, 3, 3, 1)

#     @nn.compact
#     def __call__(self, x):
#         B, H, W, C = x.shape
#         out_ch = self.out_ch if self.out_ch else C
#         if not self.fir:
#             if self.with_conv:
#                 x = conv3x3(x, out_ch, stride=2)
#             else:
#                 x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
#         else:
#             if not self.with_conv:
#                 x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
#             else:
#                 x = up_or_down_sampling.Conv2d(
#                     out_ch,
#                     kernel=3,
#                     down=True,
#                     resample_kernel=self.fir_kernel,
#                     use_bias=True,
#                     kernel_init=default_init(),
#                 )(x)

#         assert x.shape == (B, H // 2, W // 2, out_ch)
#         return x

class Downsample(nn.Module):

    def __init__(self,
        in_planes,
        out_ch=None,
        with_conv=False,
        fir=False,
        fir_kernel=(1, 3, 3, 1),
        rngs=None
    ):
        self.in_planes = in_planes
        self.out_ch = in_planes if out_ch is None else out_ch
        self.with_conv = with_conv
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.rngs = rngs

        if (not self.fir) and self.with_conv:
            self.conv3x3 = conv3x3(in_planes, self.out_ch, stride=2, rngs=self.rngs)
        elif self.with_conv:
            self.conv3x3 = up_or_down_sampling.Conv2d(
                in_dim=in_planes,
                fmaps=self.out_ch,
                kernel=3,
                down=True,
                resample_kernel=self.fir_kernel,
                use_bias=True,
                kernel_init=default_init(),
                rngs=self.rngs
            )

    def __call__(self, x):
        B, H, W, C = x.shape
        # print("in Downsample")
        # print(x.shape)
        # print(self.in_planes)
        assert self.in_planes == C
        if not self.fir:
            if self.with_conv:
                x = self.conv3x3(x)
            else:
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        else:
            if not self.with_conv:
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = self.conv3x3(x)

        assert x.shape == (B, H // 2, W // 2, self.out_ch)
        return x


# class ResnetBlockDDPMpp(nn.Module):
#     """ResBlock adapted from DDPM."""

#     act: Any
#     out_ch: Optional[int] = None
#     conv_shortcut: bool = False
#     dropout: float = 0.1
#     skip_rescale: bool = False
#     init_scale: float = 0.0

#     @nn.compact
#     def __call__(self, x, temb=None, train=True):
#         B, H, W, C = x.shape
#         out_ch = self.out_ch if self.out_ch else C
#         h = self.act(nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x))
#         h = conv3x3(h, out_ch)
#         # Add bias to each feature map conditioned on the time embedding
#         if temb is not None:
#             h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[
#                 :, None, None, :
#             ]

#         h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
#         h = nn.Dropout(self.dropout)(h, deterministic=not train)
#         h = conv3x3(h, out_ch, init_scale=self.init_scale)
#         if C != out_ch:
#             if self.conv_shortcut:
#                 x = conv3x3(x, out_ch)
#             else:
#                 x = NIN(out_ch)(x)

#         if not self.skip_rescale:
#             return x + h
#         else:
#             return (x + h) / np.sqrt(2.0)
        
class ResnetBlockDDPMpp(nn.Module):
    """ResBlock adapted from DDPM."""

    def __init__(self,
        in_planes,
        act,
        out_ch=None,
        conv_shortcut=False,
        dropout=0.1,
        skip_rescale=False,
        init_scale=0.0,
        temb_dim=None,
        rngs=None
    ):
        self.in_planes = in_planes
        self.act = act
        self.out_ch = out_ch if out_ch is not None else in_planes
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.skip_rescale = skip_rescale
        self.init_scale = init_scale
        self.temb_dim = temb_dim
        self.rngs = rngs

        self.group_norm1 = nn.GroupNorm(num_groups=min(self.in_planes // 4, 32), rngs=self.rngs)

        self.conv1 = conv3x3(self.in_planes, self.out_ch, rngs=self.rngs)

        if self.temb_dim is not None:
            self.temb_linear = nn.Linear(self.temb_dim, self.out_ch, kernel_init=default_init(), rngs=self.rngs)

        self.group_norm2 = nn.GroupNorm(num_groups=min(self.out_ch // 4, 32), rngs=self.rngs)

        self.dropout = nn.Dropout(self.dropout, rngs=self.rngs)

        self.conv2 = conv3x3(self.out_ch, self.out_ch, init_scale=self.init_scale, rngs=self.rngs)

        if self.out_ch != self.in_planes:
            if self.conv_shortcut:
                self.conv_shortcut = conv3x3(self.in_planes, self.out_ch, rngs=self.rngs)
            else:
                self.conv_shortcut = NIN(self.in_planes, self.out_ch, rngs=self.rngs)

    def __call__(self, x, temb=None, train=True):
        B, H, W, C = x.shape
        assert C == self.in_planes
        h = self.act(self.group_norm1(x))
        h = self.conv1(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            assert temb.shape == (B, self.temb_dim)
            h += self.temb_linear(self.act(temb))[
                :, None, None, :
            ]

        assert h.shape == (B, H, W, self.out_ch)
        h = self.act(self.group_norm2(h))
        h = self.dropout(h, deterministic=not train)
        h = self.conv2(h)
        if C != self.out_ch:
            x = self.conv_shortcut(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


# class ResnetBlockBigGANpp(nn.Module):
#     """ResBlock adapted from BigGAN."""

#     act: Any
#     up: bool = False
#     down: bool = False
#     out_ch: Optional[int] = None
#     dropout: float = 0.1
#     fir: bool = False
#     fir_kernel: Tuple[int] = (1, 3, 3, 1)
#     skip_rescale: bool = True
#     init_scale: float = 0.0

#     @nn.compact
#     def __call__(self, x, temb=None, train=True):
#         B, H, W, C = x.shape
#         out_ch = self.out_ch if self.out_ch else C
#         h = self.act(nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x))

#         if self.up:
#             if self.fir:
#                 h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
#                 x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
#             else:
#                 h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
#                 x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
#         elif self.down:
#             if self.fir:
#                 h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
#                 x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
#             else:
#                 h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
#                 x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

#         h = conv3x3(h, out_ch)
#         # Add bias to each feature map conditioned on the time embedding
#         if temb is not None:
#             h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[
#                 :, None, None, :
#             ]
#         h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
#         h = nn.Dropout(self.dropout)(h, deterministic=not train)
#         h = conv3x3(h, out_ch, init_scale=self.init_scale)
#         if C != out_ch or self.up or self.down:
#             x = conv1x1(x, out_ch)

#         if not self.skip_rescale:
#             return x + h
#         else:
#             return (x + h) / np.sqrt(2.0)

class ResnetBlockBigGANpp(nn.Module):
    """ResBlock adapted from BigGAN."""

    def __init__(self,
        in_planes,
        act,
        up=False,
        down=False,
        out_ch=None,
        dropout=0.1,
        fir=False,
        fir_kernel=(1, 3, 3, 1),
        skip_rescale=True,
        init_scale=0.0,
        temb_dim=None,
        rngs=None
    ):
        self.in_planes = in_planes
        self.act = act
        self.up = up
        self.down = down
        self.out_ch = out_ch if out_ch is not None else in_planes
        self.dropout = dropout
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale
        self.init_scale = init_scale
        self.temb_dim = temb_dim
        self.rngs = rngs

        self.group_norm1 = nn.GroupNorm(num_features=in_planes, num_groups=min(self.in_planes // 4, 32), rngs=self.rngs)

        self.conv1 = conv3x3(self.in_planes, self.out_ch, rngs=self.rngs)

        if self.temb_dim is not None:
            self.temb_linear = nn.Linear(self.temb_dim, self.out_ch, kernel_init=default_init(), rngs=self.rngs)

        self.group_norm2 = nn.GroupNorm(num_features=self.out_ch, num_groups=min(self.out_ch // 4, 32), rngs=self.rngs)

        self.dropout = nn.Dropout(self.dropout, rngs=self.rngs)

        self.conv2 = conv3x3(self.out_ch, self.out_ch, init_scale=self.init_scale, rngs=self.rngs)

        if self.out_ch != self.in_planes or self.up or self.down:
            self.conv1x1 = conv1x1(self.in_planes, self.out_ch, rngs=self.rngs)


    def __call__(self, x, temb=None, train=True):
        B, H, W, C = x.shape
        assert C == self.in_planes
        h = self.act(self.group_norm1(x))

        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

        h = self.conv1(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            assert temb.shape == (B, self.temb_dim)
            h += self.temb_linear(self.act(temb))[
                :, None, None, :
            ]
        h = self.act(self.group_norm2(h))
        h = self.dropout(h, deterministic=not train)
        h = self.conv2(h)
        if C != self.out_ch or self.up or self.down:
            x = self.conv1x1(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)
