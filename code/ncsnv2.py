import flax.nnx as nn
import jax.numpy as jnp
import zhh.F as F
from functools import partial
from zhh.debug import print_stat, print_tensor, set_debug
import jax
import optax
import math
import flax

from zhh.models import ModuleWrapper, TorchLinear

from utils.utils import get_sigmas
from layers import *
from normalization import get_normalization

from 助教模型 import UNet

知道=NotADirectoryError
可能是list也可能是jnparray=知道

class Sigmas(nn.Variable): pass

class NCSNv2(nn.Module):
    def __init__(self,
        ngf: int,
        n_noise_levels: int,
        logit_transform: bool=False,
        rescaled: bool=False,
        config: dict={},
        dtype: jnp.dtype = jnp.float32,
        rngs=None,
    ):
        self.ngf = ngf
        self.n_noise_levels = n_noise_levels
        self.logit_transform = logit_transform
        self.rescaled = rescaled
        self.config = config
        self.dtype = dtype
        self.rngs = rngs
        norm = get_normalization(self.config.model, conditional=False)
        data_channels = config.dataset.channels

        self.activation = activation = get_act(self.config.model)
        self.sigmas = get_sigmas(self.config.sampling)
        self.sigmas = self.sigmas.astype(self.dtype)
        self.sigmas = Sigmas(self.sigmas)

        # self.begin_conv = nn.Conv(features=ngf, kernel_size=(3, 3), padding=1, strides=1)
        self.begin_conv = nn.Conv(data_channels, ngf, kernel_size=(3, 3), padding="SAME", strides=1, rngs=rngs)
        # TODO: check the initialization of conv

        self.normalizer = norm(ngf, rngs=rngs)
        # self.end_conv = nn.Conv(data_channels, kernel_size=(3, 3), padding=1, strides=1)
        self.end_conv = nn.Conv(ngf, data_channels, kernel_size=(3, 3), padding="SAME", strides=1, rngs=rngs)

        # self.res1 = nn.ModuleList([
        #     ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
        #                   normalization=self.norm),
        #     ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
        #                   normalization=self.norm)]
        # )
        self.res1 = nn.Sequential(
            ResidualBlock(ngf, ngf, resample=None, act=activation, normalization=norm, rngs=rngs),
            ResidualBlock(ngf, ngf, resample=None, act=activation, normalization=norm, rngs=rngs)
        )

        # self.res2 = nn.ModuleList([
        #     ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
        #                   normalization=self.norm),
        #     ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
        #                   normalization=self.norm)]
        # )
        self.res2 = nn.Sequential(
            ResidualBlock(ngf, 2 * ngf, resample='down', act=activation, normalization=norm, rngs=rngs),
            ResidualBlock(2 * ngf, 2 * ngf, resample=None, act=activation, normalization=norm, rngs=rngs),
        )

        # self.res3 = nn.ModuleList([
        #     ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
        #                   normalization=self.norm, dilation=2),
        #     ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
        #                   normalization=self.norm, dilation=2)]
        # )

        self.res3 = nn.Sequential(
            ResidualBlock(2 * ngf, 2 * ngf, resample='down', act=activation, normalization=norm, dilation=2, rngs=rngs),
            ResidualBlock(2 * ngf, 2 * ngf, resample=None, act=activation, normalization=norm, dilation=2, rngs=rngs),
        )

        if config.dataset.image_size == 28:
            # self.res4 = nn.ModuleList([
            #     ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
            #                   normalization=self.norm, adjust_padding=True, dilation=4),
            #     ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
            #                   normalization=self.norm, dilation=4)]
            # )
            self.res4 = nn.Sequential(
                ResidualBlock(2 * ngf, 2 * ngf, resample='down', act=activation, normalization=norm, adjust_padding=True, dilation=4, rngs=rngs),
                ResidualBlock(2 * ngf, 2 * ngf, resample=None, act=activation, normalization=norm, dilation=4, rngs=rngs),
            )
        else:
            # self.res4 = nn.ModuleList([
            #     ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
            #                   normalization=self.norm, adjust_padding=False, dilation=4),
            #     ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
            #                   normalization=self.norm, dilation=4)]
            # )
            self.res4 = nn.Sequential(
                ResidualBlock(2 * ngf, 2 * ngf, resample='down', act=activation, normalization=norm, adjust_padding=False, dilation=4, rngs=rngs),
                ResidualBlock(2 * ngf, 2 * ngf, resample=None, act=activation, normalization=norm, dilation=4, rngs=rngs),
            )

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=activation, start=True, rngs=rngs)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=activation, rngs=rngs)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=activation, rngs=rngs)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=activation, end=True, rngs=rngs)

    # def _compute_cond_module(self, module, x):
    #     for m in module:
    #         x = m(x)
    #     return x

    def __call__(self, x, y):
        # import time
        # start = time.time()
        # def show(s): print (s,"Time: ", time.time() - start)
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self.res1(output) # shape (bs, 28, 28, ngf)
        # show("layer1")
        # print("layer1 square mean: ", torch.mean(output ** 2))
        layer2 = self.res2(layer1) # shape (bs, 14, 14, 2ngf)
        layer3 = self.res3(layer2)
        # show("layer3")
        # print("layer3 square mean: ", torch.mean(layer3 ** 2))
        layer4 = self.res4(layer3) # shape (bs, 14, 14, 2ngf)

        # print("layer1 shape: ", layer1.shape, flush=True)
        # print("layer2 shape: ", layer2.shape, flush=True)
        # print("layer3 shape: ", layer3.shape, flush=True)
        # print("layer4 shape: ", layer4.shape, flush=True)
        # show("layer4")
        # print("layer4 square mean: ", torch.mean(layer4 ** 2))

        ref1 = self.refine1([layer4], layer4.shape[1:3]) # shape (bs, 14, 14, 2ngf)
        ref2 = self.refine2([layer3, ref1], layer3.shape[1:3]) # shape (bs, 14, 14, 2ngf)
        # show("ref2")
        # print("ref2 square mean: ", torch.mean(ref2 ** 2)) 

        ref3 = self.refine3([layer2, ref2], layer2.shape[1:3]) # shape (bs, 14, 14, ngf)
        output = self.refine4([layer1, ref3], layer1.shape[1:3]) # shape (bs, 28, 28, ngf)
        # show("ref4")
        # print("ref4 square mean: ", torch.mean(output ** 2))

        # print("ref1 shape: ", ref1.shape, flush=True)
        # print("ref2 shape: ", ref2.shape, flush=True)
        # print("ref3 shape: ", ref3.shape, flush=True)
        # print("output (ref4) shape: ", output.shape, flush=True)

        output = self.normalizer(output)
        output = self.activation(output)
        output = self.end_conv(output)
        # show("end_conv")
        # print("end_conv square mean: ", torch.mean(output ** 2))

        used_sigmas = self.sigmas[y].reshape(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas
        # show('end')
        # print("output square mean: ", torch.mean(output ** 2))
        return output


class NCSNv2Deeper(nn.Module):
    def __init__(self, config):
        raise NotImplementedError("NCSNv2Deeper is not implemented yet!")
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False)
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = get_act(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        self.res5 = nn.ModuleList([
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=4),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([4 * self.ngf], 4 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine4 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine5 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)
        layer5 = self._compute_cond_module(self.res5, layer4)

        ref1 = self.refine1([layer5], layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
        ref3 = self.refine3([layer3, ref2], layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
        output = self.refine5([layer1, ref4], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output


class NCSNv2Deepest(nn.Module):
    def __init__(self, config):
        raise NotImplementedError("NCSNv2Deepest is not implemented yet!")
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False)
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = get_act(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res31 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        self.res5 = nn.ModuleList([
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=4),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([4 * self.ngf], 4 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine31 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine4 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine5 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer31 = self._compute_cond_module(self.res31, layer3)
        layer4 = self._compute_cond_module(self.res4, layer31)
        layer5 = self._compute_cond_module(self.res5, layer4)

        ref1 = self.refine1([layer5], layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
        ref31 = self.refine31([layer31, ref2], layer31.shape[2:])
        ref3 = self.refine3([layer3, ref31], layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
        output = self.refine5([layer1, ref4], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output

NCSNv2_base = partial(
    NCSNv2,
    logit_transform=False,
    rescaled=False,
)

助教模型 = UNet