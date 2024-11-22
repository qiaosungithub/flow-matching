# SQA's nnx implementation of NCSNv2 repo's normalization.py

import flax.nnx as nn
import jax.numpy as jnp
import flax
import jax


def get_normalization(config, conditional=True):
    """
    config: the model config
    """
    norm = config.normalization
    if conditional:
        if norm == 'NoneNorm':
            return ConditionalNoneNorm2d
        elif norm == 'InstanceNorm++':
            return ConditionalInstanceNorm2dPlus
        elif norm == 'InstanceNorm':
            return ConditionalInstanceNorm2d
        elif norm == 'BatchNorm':
            return ConditionalBatchNorm2d
        elif norm == 'VarianceNorm':
            return ConditionalVarianceNorm2d
        else:
            raise NotImplementedError("{} does not exist!".format(norm))
    else:
        if norm == 'BatchNorm':
            return nn.BatchNorm2d
        elif norm == 'InstanceNorm':
            return nn.InstanceNorm2d
        elif norm == 'InstanceNorm++':
            return InstanceNorm2dPlus
        elif norm == 'VarianceNorm':
            return VarianceNorm2d
        elif norm == 'NoneNorm':
            return NoneNorm2d
        elif norm is None:
            return None
        else:
            raise NotImplementedError("{} does not exist!".format(norm))

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        raise NotImplementedError("ConditionalBatchNorm2d is not implemented yet!")
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out


class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        raise NotImplementedError("ConditionalInstanceNorm2d is not implemented yet!")
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        h = self.instance_norm(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalVarianceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, bias=False):
        raise NotImplementedError("ConditionalVarianceNorm2d is not implemented yet!")
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.embed = nn.Embedding(num_classes, num_features)
        self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        vars = torch.var(x, dim=(2, 3), keepdim=True)
        h = x / torch.sqrt(vars + 1e-5)

        gamma = self.embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class VarianceNorm2d(nn.Module):
    def __init__(self, num_features, bias=False):
        raise NotImplementedError("VarianceNorm2d is not implemented yet!")
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)

    def forward(self, x):
        vars = torch.var(x, dim=(2, 3), keepdim=True)
        h = x / torch.sqrt(vars + 1e-5)

        out = self.alpha.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalNoneNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        raise NotImplementedError("ConditionalNoneNorm2d is not implemented yet!")
        self.num_features = num_features
        self.bias = bias
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * x + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * x
        return out


class NoneNorm2d(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()
        raise NotImplementedError("NoneNorm2d is not implemented yet!")

    def forward(self, x):
        return x

# class InstanceNorm2d(nn.Module):
#     """
#     Instance Normalization: norm on the h, w dim
#     """
#     def __init__(self, use_bias=False, use_scale=False, epsilon=1e-5):
#         super().__init__()
#         assert not use_bias and not use_scale, "InstanceNorm2d does not support bias and scale parameters"
#         self.eps = epsilon
#     def __call__(self, x):
#         mean = jnp.mean(x, axis=(2, 3), keepdims=True)
#         var = jnp.var(x, axis=(2, 3), keepdims=True)
#         x = (x - mean) / jnp.sqrt(var + self.eps)
#         return x

class InstanceNorm2dPlus(nn.Module):
    def __init__(self,
        num_features: int,
        bias: bool=True,
        rngs=None
    ):
        self.num_features = num_features
        self.bias = bias
        self.rngs = rngs
        # self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        # self.instance_norm = flax.linen.InstanceNorm(use_bias=False, use_scale=False, epsilon=1e-5)
        # self.alpha = nn.Parameter(jnp.zeros(num_features))
        # self.gamma = nn.Parameter(jnp.zeros(num_features))
        # self.alpha = self.param('alpha', nn.initializers.normal(stddev=0.02), (num_features,))
        # self.gamma = self.param('gamma', nn.initializers.normal(stddev=0.02), (num_features,))
        self.alpha = nn.Embed(num_embeddings=1, features=num_features, embedding_init=nn.initializers.normal(stddev=0.02), rngs=rngs)
        self.gamma = nn.Embed(num_embeddings=1, features=num_features, embedding_init=nn.initializers.normal(stddev=0.02), rngs=rngs)
        # self.alpha.data.normal_(1, 0.02)
        # self.gamma.data.normal_(1, 0.02)
        if bias:
            # self.beta = nn.Parameter(jnp.zeros(num_features))
            # self.beta = self.param('beta', nn.initializers.zeros, (num_features,))
            self.beta = nn.Embed(num_embeddings=1, features=num_features, embedding_init=nn.initializers.zeros, rngs=rngs)

    def __call__(self, x):
        bs = x.shape[0] # here the image has shape (bs, h, w, c) in jnp style
        # print("x.shape", x.shape, flush=True)
        x = x.transpose((0, 3, 1, 2)) # (bs, c, h, w)
        means = jnp.mean(x, axis=(2, 3)) 
        var = jnp.var(x, axis=(2, 3))
        m = jnp.mean(means, axis=-1, keepdims=True)
        v = jnp.var(means, axis=-1, keepdims=True)
        h = (x - means[..., None, None]) / (jnp.sqrt(var[..., None, None] + 1e-5))
        means = (means - m) / (jnp.sqrt(v + 1e-5))
        # h = self.instance_norm(x)
        # print("means.shape", means.shape, flush=True)
        # print("h.shape", h.shape, flush=True)

        if self.bias:
            h = h + means[..., None, None] * self.alpha(jnp.zeros(bs,dtype=jnp.int32))[..., None, None]
            out = self.gamma(jnp.zeros(bs,dtype=jnp.int32)).reshape(-1, self.num_features, 1, 1) * h + self.beta(jnp.zeros(bs,dtype=jnp.int32)).reshape(-1, self.num_features, 1, 1)
        else:
            h = h + means[..., None, None] * self.alpha(jnp.zeros(bs,dtype=jnp.int32))[..., None, None]
            out = self.gamma(jnp.zeros(bs,dtype=jnp.int32)).reshape(-1, self.num_features, 1, 1) * h
        # reshape back to (bs, h, w, c)
        out = out.transpose((0, 2, 3, 1))
        return out


class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        raise NotImplementedError("ConditionalInstanceNorm2dPlus is not implemented yet!")
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out
