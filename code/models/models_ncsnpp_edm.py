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

from .jcm import layers, layerspp, normalization
# from jcm import layers, layerspp, normalization
# import flax.linen as nn
import flax.nnx as nn
import functools
from functools import partial
import jax.numpy as jnp
import jax
import numpy as np
import ml_collections

from typing import Any, Sequence


from absl import logging

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine # not used
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1 # not used
get_act = layers.get_act # not used
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self,
        base_width = 128,
        image_size = 32,
        out_channels = 3, # also in_channels
        ch_mult = (2, 2, 2),
        num_res_blocks = 4,
        attn_resolutions = (16,),
        dropout = 0.0,
        fir_kernel = (1, 3, 3, 1),
        resblock_type = "biggan",
        embedding_type = "fourier",
        fourier_scale = 16.0,
        rngs = None,
        use_aug_label = False,
        aug_label_dim = None,
        **kwargs
    ):

        nf = self.base_width = base_width
        self.image_size = image_size
        self.out_channels = out_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.fir_kernel = fir_kernel
        self.resblock_type = resblock_type
        self.embedding_type = embedding_type
        # assert embedding_type == "positional", '其实搞对了'
        self.fourier_scale = fourier_scale
        self.rngs = rngs
        self.use_aug_label = use_aug_label
        self.aug_label_dim = aug_label_dim

        self.act = act = nn.swish
        self.init_scale = init_scale = 0.0
        self.skip_rescale = skip_rescale = True
        self.resamp_with_conv = resamp_with_conv = True
        self.fir = fir = True
        self.double_heads = double_heads = False

        cur_size = image_size
        self.num_resolutions = num_resolutions = len(ch_mult)

        progressive = self.progressive = "none"
        progressive_input = self.progressive_input = "residual"
        
        assert progressive in ["none", "output_skip", "residual"]
        assert self.progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]

        ################ time embedding layer ################
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            self.temb_layer = layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=fourier_scale, rngs=rngs
            )
        elif embedding_type == "positional":
            # Sinusoidal positional embeddings.
            self.temb_layer = partial(layers.get_timestep_embedding, embedding_dim=nf)
        else:
            raise NotImplementedError
            raise ValueError(f"embedding type {embedding_type} unknown.")
        self.input_temb_dim = input_temb_dim = nf if embedding_type == "positional" else 2 * nf # NOTE: here, if use fourier embedding, the output dim is 2 * nf; for positional embedding, the output dim is nf. This is tang
        #################### aug label ############################
        assert not use_aug_label
        if use_aug_label:
            assert 'assert' == exec('assert "assert"') # 你这其实就能进来
            assert aug_label_dim is not None
            assert embedding_type == "positional" # in edm_jax, Kaiming only supports positional embedding
            self.augemb_layer = nn.Linear(aug_label_dim, input_temb_dim, kernel_init=default_initializer(), use_bias=False, rngs=rngs)
        #################### noise condition ############################
        input_temb_dim = self.input_temb_dim
        self.cond_MLP = nn.Sequential(
            nn.Linear(input_temb_dim, nf * 4, kernel_init=default_initializer(), rngs=rngs),
            act,
            nn.Linear(nf * 4, nf * 4, kernel_init=default_initializer(), rngs=rngs),
        )
        #################### Blocks ############################
        
        AttnBlock = partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale, rngs=rngs
        )

        Upsample = partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
            rngs=rngs,
        )

        # Downsample = partial(
        #     layerspp.Downsample,
        #     with_conv=resamp_with_conv,
        #     fir=fir,
        #     fir_kernel=fir_kernel,
        #     rngs=rngs,
        # )
        #################### progressive (input) #########################
        if progressive == "output_skip":
            raise NotImplementedError
            pyramid_upsample = partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            raise NotImplementedError
            pyramid_upsample = partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if progressive_input == "input_skip":
            raise NotImplementedError
            pyramid_downsample = partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            # TODO: what is in and out shape here?
            self.pyramid_downsample = partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True, rngs=rngs
            )
        #################### resblock #########################
        if resblock_type == "ddpm":
            raise NotImplementedError
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                rngs=rngs,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")
        #################### blocks #########################
        c_list = []
        setattr(self, f'enc_{cur_size}x{cur_size}_conv', conv3x3(out_channels, nf, rngs=rngs))
        c_list.append(nf)
        # downsample
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                out_c = nf * ch_mult[i_level]
                in_c = out_c if i_block > 0 else (nf * (1 if i_level == 0 else ch_mult[i_level - 1]))
                setattr(
                    self,
                    f'enc_{cur_size}x{cur_size}_block{i_block}',
                    ResnetBlock(in_c, out_ch=out_c, temb_dim=4*nf),
                )
                if cur_size in attn_resolutions:
                    setattr(
                        self,
                        f'enc_{cur_size}x{cur_size}_block{i_block}_attn',
                        AttnBlock(out_c),
                    )
                c_list.append(out_c)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    raise NotImplementedError
                    setattr(self, f'enc_{cur_size}x{cur_size}_down', Downsample())
                else:
                    cur_size //= 2
                    setattr(
                        self,
                        f'enc_{cur_size}x{cur_size}_down',
                        ResnetBlock(out_c, down=True, temb_dim=4*nf),
                    )

                if self.progressive_input == "input_skip":
                    raise NotImplementedError
                elif self.progressive_input == "residual":
                    in_dim = nf * ch_mult[i_level-1] if i_level > 0 else out_channels
                    setattr(
                        self,
                        f'enc_{cur_size}x{cur_size}_aux_residual',
                        self.pyramid_downsample(in_planes=in_dim, out_ch=out_c),
                    )
                c_list.append(out_c)
        
        c = nf * ch_mult[-1]
        # middle
        setattr(self, 
                f'dec_{cur_size}x{cur_size}_in0', 
                ResnetBlock(c, temb_dim=4*nf)
        )
        setattr(self, 
                f'dec_{cur_size}x{cur_size}_in0_attn', 
                AttnBlock(c)
        )
        setattr(self,
                f'dec_{cur_size}x{cur_size}_in1',
                ResnetBlock(c, temb_dim=4*nf)
        )
        # upsample
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_c = nf * ch_mult[i_level]
                in_c = c_list.pop() + (out_c if i_block > 0 else (nf * (ch_mult[i_level+1] if i_level < num_resolutions - 1 else ch_mult[-1])))
                # this is 恶臭
                setattr(
                    self,
                    f'dec_{cur_size}x{cur_size}_block{i_block}',
                    ResnetBlock(in_c, out_ch=out_c, temb_dim=4 * nf),
                )
            if cur_size in attn_resolutions:
                setattr(
                    self,
                    f'dec_{cur_size}x{cur_size}_block{i_block}_attn',
                    AttnBlock(out_c),
                )
            if progressive != "none":
                raise NotImplementedError
            if i_level != 0:
                if resblock_type == "ddpm":
                    raise NotImplementedError
                    setattr(self, f'dec_{cur_size}x{cur_size}_up', Upsample())
                else:
                    cur_size *= 2
                    setattr(
                        self,
                        f'dec_{cur_size}x{cur_size}_up',
                        ResnetBlock(out_c, up=True, temb_dim=4*nf),
                    )
        assert not c_list
        # final
        if self.progressive == "output_skip" and not double_heads:
            raise NotImplementedError
        else: 
            in_c = nf * ch_mult[0]
            setattr(self, 
                    f'dec_{cur_size}x{cur_size}_aux_norm', 
                    nn.GroupNorm(num_features=in_c, num_groups=min(in_c // 4, 32), rngs=rngs)
            )
            if double_heads:
                raise NotImplementedError
            else:
                setattr(self, 
                        f'dec_{cur_size}x{cur_size}_aux_conv', 
                        conv3x3(in_c, out_channels, init_scale=init_scale, rngs=rngs)
                )

    def __call__(self, x, time_cond, augment_label=None, train=True, verbose=False): # turn off verbose here

        # print("in call of ncsnpp model")
        # print("x.shape", x.shape)
        # print("time_cond.shape", time_cond.shape)
        assert time_cond.ndim == 1  # only support 1-d time condition
        assert time_cond.shape[0] == x.shape[0]
        assert x.shape[-1] == self.out_channels

        logging_fn = logging.info if verbose else lambda x: None

        # --------------------
        # redefine arguments:
        act = nn.swish

        nf = self.base_width  # config.model.nf
        ch_mult = self.ch_mult
        num_res_blocks = self.num_res_blocks
        attn_resolutions = self.attn_resolutions
        # resamp_with_conv = True # not used
        num_resolutions = len(ch_mult)

        skip_rescale = True
        init_scale = 0.0

        # combiner = functools.partial(Combine, method=combine_method)
        combiner = None # not used

        # --------------------

        # timestep/noise_level embedding; only for continuous training
        temb = self.temb_layer(time_cond)
        assert temb.shape[-1] == self.input_temb_dim

        if augment_label is not None:
            assert self.use_aug_label
            assert augment_label.shape == (x.shape[0], self.aug_label_dim)
            aemb = self.augemb_layer(augment_label)
            temb += aemb 

        temb = self.cond_MLP(temb)

        # utility function to count number of parameters
        def pms(self, name):
            """
            output: number of parameters
            """
            layer = getattr(self, name)
            tree = jax.tree.map(lambda x: np.prod(x.shape), nn.state(layer))
            return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0)
        ps = partial(pms, self)

        ########### begin of the work ############
        # Downsampling block

        cur_size = self.image_size

        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x

        logging_fn(f"Input shape {x.shape}")
        name = f'enc_{cur_size}x{cur_size}_conv'  # 32x32_conv
        hs = [getattr(self, name)(x)]
        logging_fn(f"{name}: params {ps(name)}, shape {hs[-1].shape}") # 32x32xnf
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                name = f'enc_{cur_size}x{cur_size}_block{i_block}'
                assert temb.shape[-1] == 4 * nf
                h = getattr(self, name)(hs[-1], temb, train)
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")
                # Here out_c is nf * ch_mult[i_level]
                assert h.shape[1] == cur_size
                if h.shape[1] in attn_resolutions:
                    name = f'enc_{cur_size}x{cur_size}_block{i_block}_attn'
                    h = getattr(self, name)(h)
                    logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")
                hs.append(h)

            if i_level != num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    raise NotImplementedError
                    h = Downsample()(hs[-1])
                else:
                    # downsample
                    cur_size //= 2
                    name = f'enc_{cur_size}x{cur_size}_down'
                    h = getattr(self, name)(hs[-1], temb, train)
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

                if self.progressive_input == "input_skip":
                    raise NotImplementedError
                    input_pyramid = pyramid_downsample()(input_pyramid)
                    h = combiner()(input_pyramid, h)

                elif self.progressive_input == "residual":
                    name = f'enc_{cur_size}x{cur_size}_aux_residual'
                    # print("input_pyramid.shape", input_pyramid.shape)
                    input_pyramid = getattr(self, name)(input_pyramid)
                    if skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(
                            2.0, dtype=np.float32
                        )
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                    logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

                # logging_fn(f"Level {i_level}, combined shape {h.shape}")
                hs.append(h)

        # cnt=0
        # for h in hs:
        #     print("layer", cnt)
        #     cnt+=1
        #     print(f"shape: {h.shape}")
        #     print(f"sum: {jnp.sum(h**2)}")

        h = hs[-1]
        assert h.shape[-1] == nf * ch_mult[-1]
        name = f'dec_{cur_size}x{cur_size}_in0'
        h = getattr(self, name)(h, temb, train)
        logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")
        
        name = f'dec_{cur_size}x{cur_size}_in0_attn'
        h = getattr(self, name)(h)
        logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")
        
        name = f'dec_{cur_size}x{cur_size}_in1'
        h = getattr(self, name)(h, temb, train)
        logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

        pyramid = None

        # print(f"shape: {h.shape}")
        # print(f"sum: {jnp.sum(h**2)}")

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                name = f'dec_{cur_size}x{cur_size}_block{i_block}'
                h = getattr(self, name)(
                    jnp.concatenate([h, hs.pop()], axis=-1), temb, train
                )
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

            assert h.shape[1] == cur_size
            if h.shape[1] in attn_resolutions:
                name = f'dec_{cur_size}x{cur_size}_block{i_block}_attn'
                h = getattr(self, name)(h)
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

            if self.progressive != "none":
                raise NotImplementedError
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        pyramid = conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            x.shape[-1],
                            bias=True,
                            init_scale=init_scale,
                        )
                    elif progressive == "residual":
                        pyramid = conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            h.shape[-1],
                            bias=True,
                        )
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        pyramid = pyramid_upsample()(pyramid)
                        pyramid = pyramid + conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            x.shape[-1],
                            bias=True,
                            init_scale=init_scale,
                        )
                    elif progressive == "residual":
                        pyramid = pyramid_upsample(out_ch=h.shape[-1])(pyramid)
                        if skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0, dtype=np.float32)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    raise NotImplementedError
                    h = Upsample()(h)
                else:
                    cur_size *= 2
                    name = f'dec_{cur_size}x{cur_size}_up'
                    h = getattr(self, name)(h, temb, train)
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

            # print(f"shape: {h.shape}")
            # print(f"sum: {jnp.sum(h**2)}")

        assert not hs

        if self.progressive == "output_skip" and not self.double_heads:
            raise NotImplementedError
            h = pyramid
        else:
            assert h.shape[-1] == nf * ch_mult[0]
            name = f'dec_{cur_size}x{cur_size}_aux_norm'
            h = act(getattr(self, name)(h))
            logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")
            if self.double_heads:
                raise NotImplementedError
                h = conv3x3(h, x.shape[-1] * 2, init_scale=init_scale)
            else:
                name = f'dec_{cur_size}x{cur_size}_aux_conv'
                h = getattr(self, name)(h)
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")
        logging_fn(f"Output shape {h.shape}")
        return h


# # test

# rngs = nn.Rngs(0, params=114, dropout=514, train=1919)
# model = NCSNpp(base_width=16, rngs=rngs)
# from jax import random
# inputs = random.normal(rngs.train(), (2, 32, 32, 3))
# time_cond = jnp.log(jnp.array([1, 0.1]))
# output = model(inputs, time_cond, train=True, verbose=True)
# print(output.shape)
# print(jnp.sum(output**2))