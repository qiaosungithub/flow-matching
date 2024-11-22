# SQA's nnx implementation of NCSNv2 repo's layers.py

import flax.nnx as nn
import flax
import zhh.F as F
from functools import partial
import math
import jax.numpy as jnp
import jax

from models.normalization import *

知道 = ValueError

def get_act(config):
    """
    config: the model config
    """
    if config.activation.lower() == 'elu':
        return nn.elu
    elif config.activation.lower() == 'relu':
        return nn.relu
    elif config.activation.lower() == 'lrelu':
        return partial(nn.leaky_relu, negative_slope=0.2)
    elif config.activation.lower() == 'swish':
        def swish(x):
            return x * nn.sigmoid(x)
        return swish
    else:
        raise NotImplementedError('activation function does not exist!')

# def spectral_norm(layer, n_iters=1):
#     return torch.nn.utils.spectral_norm(layer, n_power_iterations=n_iters)
def spectral_norm(layer, n_iters=1):
    raise NotImplementedError('spectral_norm is not implemented in flax')

def conv1x1(in_planes, out_planes, rngs, stride=1, bias=True, spec_norm=False):
    "1x1 convolution"
    conv = nn.Conv(in_planes, out_planes, kernel_size=(1, 1), strides=stride,
                     padding='SAME', use_bias=bias, rngs=rngs)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def conv3x3(in_planes, out_planes, rngs, stride=1, bias=True, spec_norm=False):
    "3x3 convolution with padding"
    conv = nn.Conv(in_planes, out_planes, kernel_size=(3, 3), strides=stride,
                     padding='SAME', use_bias=bias, rngs=rngs)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv


def stride_conv3x3(in_planes, out_planes, kernel_size, rngs, bias=True, spec_norm=False):
    if type(kernel_size) == int:
        kernel_size = (kernel_size, kernel_size)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    conv = nn.Conv(in_planes, out_planes, kernel_size=kernel_size, strides=2,
                     padding=padding, use_bias=bias, rngs=rngs)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def dilated_conv3x3(in_planes, out_planes, dilation, rngs, bias=True, spec_norm=False):
    conv = nn.Conv(in_planes, out_planes, kernel_size=(3, 3), padding=dilation, 
                   kernel_dilation=dilation, use_bias=bias, rngs=rngs)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv

# # Cascased Residual Blocks
# class CRPBlock(nn.Module):
#     def __init__(self, features, n_stages, act=nn.relu, maxpool=True, spec_norm=False):
#         super().__init__()
#         self.convs = nn.ModuleList()
#         for i in range(n_stages):
#             self.convs.append(conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm))
#         self.n_stages = n_stages
#         if maxpool:
#             self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
#         else:
#             self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

#         self.act = act

#     def __call__(self, x):
#         x = self.act(x)
#         path = x
#         for i in range(self.n_stages):
#             path = self.maxpool(path)
#             path = self.convs[i](path)
#             x = path + x
#         return x
    
class CRPBlock(nn.Module):

    def __init__(self,
        features: int,
        n_stages: int,
        act: nn.Module,
        maxpool: bool,
        spec_norm: bool,
        rngs=None,
    ):
        self.features = features
        self.n_stages = n_stages
        self.act = act
        self.maxpool = maxpool
        self.spec_norm = spec_norm
        self.rngs = rngs

        self.convs = []
        for i in range(n_stages):
            self.convs.append(conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm, rngs=rngs))
        if maxpool:
            # self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
            self.pool = partial(nn.max_pool, window_shape=(5, 5), strides=(1, 1), padding='SAME')
        else:
            # self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
            self.pool = partial(nn.avg_pool, window_shape=(5, 5), strides=(1, 1), padding='SAME')

        self.act = act

    def __call__(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.pool(path)
            path = self.convs[i](path)
            x = path + x
        return x


# class CondCRPBlock(nn.Module):
#     def __init__(self, features, n_stages, num_classes, normalizer, act=nn.relu, spec_norm=False):
#         raise NotImplementedError('CondCRPBlock is not implemented in flax')
#         super().__init__()
#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         self.normalizer = normalizer
#         for i in range(n_stages):
#             self.norms.append(normalizer(features, num_classes, bias=True))
#             self.convs.append(conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm))

#         self.n_stages = n_stages
#         self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
#         self.act = act

#     def __call__(self, x, y):
#         x = self.act(x)
#         path = x
#         for i in range(self.n_stages):
#             path = self.norms[i](path, y)
#             path = self.maxpool(path)
#             path = self.convs[i](path)

#             x = path + x
#         return x


# Residual Convolutional Unit
class RCUBlock(nn.Module):
    def __init__(self, 
        features: int,
        n_blocks: int,
        n_stages: int,
        act: 知道=nn.relu,
        spec_norm: bool=False,
        rngs=None,):
        self.features = features
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        self.spec_norm = spec_norm
        self.rngs = rngs

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1), conv3x3(features, features, stride=1, bias=False,
                                                                         spec_norm=spec_norm, rngs=rngs))

        self.stride = 1

    def __call__(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

            x += residual
        return x


class CondRCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.relu, spec_norm=False):
        raise NotImplementedError('CondRCUBlock is not implemented in flax')
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_norm'.format(i + 1, j + 1), normalizer(features, num_classes, bias=True))
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1),
                        conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm))

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        self.normalizer = normalizer

    def __call__(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, '{}_{}_norm'.format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

            x += residual
        return x

list或者tuple = (list, tuple)

# Multi-Scale Feature Block
class MSFBlock(nn.Module):

    def __init__(self,
        in_planes: list或者tuple,
        features: int,
        spec_norm: bool=False,
        rngs=None,
    ):
        """
        :param in_planes: tuples of input planes
        """
        self.in_planes = in_planes
        self.features = features
        self.spec_norm = spec_norm
        self.rngs = rngs

        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = []

        for i in range(len(in_planes)):
            self.convs.append(conv3x3(in_planes[i], features, stride=1, bias=True, spec_norm=spec_norm, rngs=rngs))

    def __call__(self, xs, shape):
        sums = jnp.zeros(shape=(xs[0].shape[0], *shape, self.features)) # note that here image shape is (bs, h, w, c)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            # h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            h = jax.image.resize(h, shape=(xs[0].shape[0], *shape, self.features), method='bilinear')
            sums += h
        return sums


class CondMSFBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes, normalizer, spec_norm=False):
        raise NotImplementedError('CondMSFBlock is not implemented in flax')
        """
        :param in_planes: tuples of input planes
        """
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features
        self.normalizer = normalizer

        for i in range(len(in_planes)):
            self.convs.append(conv3x3(in_planes[i], features, stride=1, bias=True, spec_norm=spec_norm))
            self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

    def __call__(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums


class RefineBlock(nn.Module):
    def __init__(self,
        in_planes: list或者tuple,
        features: int,
        act: 知道=nn.relu,
        start: bool=False,
        end: bool=False,
        maxpool: bool=True,
        spec_norm: bool=False,
        rngs=None,
    ):
        self.in_planes = in_planes
        self.features = features
        self.act = act
        self.start = start
        self.end = end
        self.maxpool = maxpool
        self.spec_norm = spec_norm
        self.rngs = rngs

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = []
        for i in range(n_blocks):
            self.adapt_convs.append(
                RCUBlock(in_planes[i], 2, 2, act, spec_norm=spec_norm, rngs=rngs)
            )

        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act, spec_norm=spec_norm, rngs=rngs)

        if not start:
            self.msf = MSFBlock(in_planes, features, spec_norm=spec_norm, rngs=rngs)

        self.crp = CRPBlock(features, 2, act, maxpool=maxpool, spec_norm=spec_norm, rngs=rngs)

    def __call__(self, xs, output_shape):
        # print("in Refine Block ________________________________")
        assert isinstance(xs, tuple) or isinstance(xs, list)
        # for x in xs: print("input shape:", x.shape)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)
            # print("output shape:", h.shape)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
            # print("MSF output shape:", h.shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)
        # print("output shape:", h.shape)

        return h



class CondRefineBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes, normalizer, act=nn.relu, start=False, end=False, spec_norm=False):
        raise NotImplementedError('CondRefineBlock is not implemented in flax')
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act, spec_norm=spec_norm)
            )

        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act, spec_norm=spec_norm)

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer, spec_norm=spec_norm)

        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act, spec_norm=spec_norm)

    def __call__(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]

        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h

int或者tuple反正是形状 = 知道

class ConvMeanPool(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        kernel_size: int或者tuple反正是形状=(3, 3),
        biases: bool=True,
        adjust_padding: bool=False,
        spec_norm: bool=False,
        rngs=None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.biases = biases
        self.adjust_padding = adjust_padding
        self.spec_norm = spec_norm
        self.rngs = rngs

        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)

        # conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        conv = nn.Conv(input_dim, output_dim, kernel_size=kernel_size, strides=1, padding='SAME', use_bias=biases, rngs=rngs)
        if spec_norm:
            conv = spectral_norm(conv)
        self.conv = conv

    def __call__(self, inputs):
        if self.adjust_padding:
            inputs = jnp.pad(inputs, ((0, 0), (0, 0), (1, 0), (1, 0)), mode='constant')
            
        output = self.conv(inputs) # note that here image shape is (bs, h, w, c)
        output = sum([output[:, ::2, ::2, :], output[:, 1::2, ::2, :],
                      output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]]) / 4.
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, spec_norm=False):
        raise NotImplementedError('MeanPoolConv is not implemented in flax')
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        if spec_norm:
            self.conv = spectral_norm(self.conv)

    def __call__(self, inputs):
        output = inputs # note that here image shape is (bs, h, w, c)
        output = sum([output[:, ::2, ::2, :], output[:, 1::2, ::2, :],
                      output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]]) / 4.
        return self.conv(output)


class UpsampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, spec_norm=False):
        raise NotImplementedError('UpsampleConv is not implemented in flax')
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        if spec_norm:
            self.conv = spectral_norm(self.conv)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def __call__(self, inputs):
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        output = self.pixelshuffle(output)
        return self.conv(output)


class ConditionalResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, resample=None, act=nn.elu,
                 normalization=ConditionalBatchNorm2d, adjust_padding=False, dilation=None, spec_norm=False):
        raise NotImplementedError('ConditionalResidualBlock is not implemented in flax')
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
            else:
                self.conv1 = conv3x3(input_dim, input_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding, spec_norm=spec_norm)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding, spec_norm=spec_norm)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = conv3x3(input_dim, output_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = conv3x3(output_dim, output_dim, spec_norm=spec_norm)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim, num_classes)


    def __call__(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class ResidualBlock(nn.Module):
    def __init__(self, 
        input_dim: int,
        output_dim: int,
        resample: bool=None,
        act: 知道=nn.elu,
        normalization: 知道=nn.BatchNorm,
        adjust_padding: bool=False,
        dilation: int=None,
        spec_norm: bool=False,
        rngs=None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.act = act
        self.normalization = normalization
        self.adjust_padding = adjust_padding
        self.dilation = dilation
        self.spec_norm = spec_norm
        self.rngs = rngs

        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation, spec_norm=spec_norm, rngs=rngs)
                self.normalize2 = normalization(input_dim, rngs=rngs)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm, rngs=rngs)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm, rngs=rngs)
            else:
                self.conv1 = conv3x3(input_dim, input_dim, spec_norm=spec_norm, rngs=rngs)
                self.normalize2 = normalization(input_dim, rngs=rngs)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding, spec_norm=spec_norm, rngs=rngs)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding, spec_norm=spec_norm, rngs=rngs)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm, rngs=rngs)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm, rngs=rngs)
                self.normalize2 = normalization(output_dim, rngs=rngs)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation, spec_norm=spec_norm, rngs=rngs)
            else:
                # conv_shortcut = nn.Conv2d ### Something wierd here.
                conv_shortcut = partial(conv1x1, spec_norm=spec_norm, rngs=rngs)
                self.conv1 = conv3x3(input_dim, output_dim, spec_norm=spec_norm, rngs=rngs)
                self.normalize2 = normalization(output_dim, rngs=rngs)
                self.conv2 = conv3x3(output_dim, output_dim, spec_norm=spec_norm, rngs=rngs)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim, rngs=rngs)


    def __call__(self, x):
        # import time
        # start = time.time()
        # def show(s): print (
        #             '\t',s,"Time: ", time.time() - start)
        output = self.normalize1(x)
        output = self.act(output)
        output = self.conv1(output)
        # show("conv1")
        # print("\t output square mean: ", (output**2).mean())
        output = self.normalize2(output)
        output = self.act(output)
        output = self.conv2(output)
        # show("conv2")
        # print("\t output square mean: ", (output**2).mean())

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output
