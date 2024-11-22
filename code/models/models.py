# ZHH's nnx based UNet model, implemented by himself

from functools import partial
from typing import Any, Callable, Sequence, Tuple

import flax.nnx as nn
import jax.numpy as jnp
from absl import logging
import jax
from pprint import pprint
import utils.display_utils as udu
class RegisteredAngle(nn.Variable): pass

# class SinousEmbeeding(nn.Module):

#     def __init__(self,dim,dtype=jnp.float32):
#         assert dim % 2 == 0, f'dim must be even, but got {dim}'
#         self.dim = dim
#         # self.angles = 10000.**(-jnp.arange(0, dim, 2).astype(jnp.float32) / dim)
#         self.dtype = dtype
#         self.angles = RegisteredAngle(10000.**(-jnp.arange(0, dim, 2).astype(dtype) / dim))

#     def __call__(self,pos):
#         mul = jnp.einsum('i,d->id', pos.astype(self.dtype), self.angles.value)
#         return jnp.concatenate((jnp.sin(mul), jnp.cos(mul)), axis=-1)

class LinearEmbedding(nn.Module):

    def __init__(self,dim,rngs,dtype=jnp.float32 ):
        self.dim = dim
        self.emb = nn.Sequential(
            nn.Linear(1, dim, dtype=dtype,rngs=rngs),
            nn.gelu,
            nn.Linear(dim, dim, dtype=dtype,rngs=rngs)
        )
        self.dtype = dtype

    def __call__(self,pos):
        return self.emb(pos.astype(self.dtype).reshape(-1,1))

class Attention(nn.Module):

    def __init__(self,dim,head,rngs, attn_dim=None,dtype=jnp.float32):
        self.head = head
        self.attn_dim = attn_dim if attn_dim is not None else dim//2
        self.head_dim = self.attn_dim // head
        self.scale = self.head_dim ** -0.5
        self.Q = nn.Conv(dim, self.attn_dim,kernel_size=(1,1), use_bias=False, rngs=rngs,dtype=dtype)
        self.K = nn.Conv(dim, self.attn_dim,kernel_size=(1,1), use_bias=False, rngs=rngs,dtype=dtype)
        self.V = nn.Conv(dim, self.attn_dim,kernel_size=(1,1), use_bias=False, rngs=rngs,dtype=dtype)
        self.out = nn.Conv(self.attn_dim, dim,kernel_size=(1,1), use_bias=False, rngs=rngs,dtype=dtype)

    def __call__(self,query, context):
        b,h,w,c = query.shape
        q = self.Q(query).reshape(b, h*w, self.head, self.head_dim)
        k = self.K(context).reshape(b, h*w, self.head, self.head_dim)
        v = self.V(context).reshape(b, h*w, self.head, self.head_dim)
        score = jnp.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = nn.softmax(score, axis=-2)
        out = jnp.einsum('bijh,bjhd->bihd', attn, v).reshape(b, h, w, self.attn_dim)
        return self.out(out)

class ScaleOnlyLayerNorm(nn.Module):

    def __init__(self,dim,rngs,dtype=jnp.float32):
        self._norm = nn.LayerNorm(num_features=dim,use_bias=False,use_scale=True,rngs=rngs,dtype=dtype)
    
    def __call__(self,x):
        return self._norm(x)

class ResBlock(nn.Module):
    
    def __init__(self,t_dim,channel,rngs,kernel=3,dtype=jnp.float32):
        self.conv1 = nn.Conv(channel,channel,kernel_size=(kernel,kernel),padding='SAME',rngs=rngs,dtype=dtype)
        self.conv2 = nn.Conv(channel,channel,kernel_size=(kernel,kernel),padding='SAME',rngs=rngs,dtype=dtype)
        self.norm = ScaleOnlyLayerNorm(channel,rngs=rngs,dtype=dtype)
        self.t_net = nn.Linear(t_dim,channel,rngs=rngs,dtype=dtype)

    def __call__(self,x,t):
        # print('\t\tinput x stats:', x.min(), x.max(), x.mean(), x.std())
        t_emb = self.t_net(t)
        xc = x
        # print('\t\txc stats:', xc.min(), xc.max(), xc.mean(), xc.std())
        x = self.conv1(x)
        x = self.norm(x)
        x = nn.relu(x + t_emb.reshape(t_emb.shape[0],1,1,t_emb.shape[1]))
        x = self.conv2(x)
        # print('\t\tx stats after conv2:', x.min(), x.max(), x.mean(), x.std())
        return x + xc

class Block(nn.Module):

    def __init__(self,in_channels,out_channels,t_dim,rngs,layer=2,kernel=3,attn=False,attn_head=4,attn_dim=None,res=False,dtype=jnp.float32):
        self.init_conv = nn.Conv(in_channels,out_channels,kernel_size=(kernel,kernel),padding='SAME',rngs=rngs,dtype=dtype)
        self.t_net = nn.Linear(t_dim,out_channels,rngs=rngs,dtype=dtype)
        self.num_layer = layer
        self.have_attn = attn
        if attn:
            self.attn = nn.Sequential(*[Attention(out_channels,attn_head,attn_dim=attn_dim,rngs=rngs,dtype=dtype) for _ in range(layer)])
            self.pre_norm = nn.Sequential(*[ScaleOnlyLayerNorm(out_channels,rngs=rngs,dtype=dtype) for _ in range(layer)])
        self.have_res = res
        if res:
            self.res = nn.Sequential(*[ResBlock(t_dim,out_channels,kernel=kernel,rngs=rngs,dtype=dtype) for _ in range(layer)])

    def __call__(self,x,t):
        # print('\tinput x stats:', x.min(), x.max(), x.mean(), x.std())
        temb = self.t_net(t)
        x = self.init_conv(x) + temb.reshape(temb.shape[0],1,1,temb.shape[1])
        for l in range(self.num_layer):
            # print(f'\tsub layer {l}')
            if self.have_attn:
                x = self.pre_norm.layers[l](x)
                x = self.attn.layers[l](x,x)
                # print(f'\tx stats after attn {l}:', x.min(), x.max(), x.mean(), x.std())
            if self.have_res:
                x = self.res.layers[l](x,t)
                # print(f'\tx stats after res {l}:', x.min(), x.max(), x.mean(), x.std())
        # print('\toutput x stats:', x.min(), x.max(), x.mean(), x.std())
        return x

AvgPool2 = partial(nn.avg_pool, window_shape=(2,2), strides=(2,2))
# def Upsample2(x):
#     after = jax.image.resize(x, (x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]), method='nearest')
#     return after

def NeuralUpsample2(channel, rngs):
    return nn.ConvTranspose(channel, channel, kernel_size=(2,2), strides=(2,2), padding='SAME', rngs=rngs)

def show_max_and_min(x, pos):
    print("At position", pos, flush=True)
    print('max:', jnp.max(x), 'min:', jnp.min(x), flush=True)

class UNet(nn.Module):

    def __init__(
        self, 
        depth,
        init_channel,
        channel_multiples,
        attns,
        attn_head,
        attn_headdim,
        rngs,
        t_dim=32,
        dtype=jnp.float32
    ):
        assert len(channel_multiples) == depth + 1, f'channel_multiples is {channel_multiples}, but should be of length {depth+1}'
        assert len(attns) == depth + 1, f'attns is {attns}, but should be of length {depth+1}'
        self.depth=depth;self.init_channel=init_channel;self.channel_multiples=channel_multiples;self.attns=attns;self.attn_head=attn_head;self.attn_headdim=attn_headdim;self.rngs=rngs;self.t_dim=t_dim;self.dtype=dtype
        self.init_conv = nn.Conv(3,init_channel,kernel_size=(3,3),padding='SAME',rngs=rngs)
        # self.t_emb = SinousEmbeeding(t_dim,dtype=dtype)
        self.t_emb = LinearEmbedding(t_dim,rngs,dtype=dtype)
        self.up = nn.Sequential(*self.get_up_layers())
        self.middle = Block(self.channel_multiples[-1]*self.init_channel,self.channel_multiples[-1]*self.init_channel,t_dim,layer=2,res=True,attn=self.attns[-1],rngs=rngs,dtype=dtype)
        self.down = nn.Sequential(*self.get_down_layers())
        self.end = nn.Conv(self.init_channel,3,kernel_size=(3,3),padding='SAME',rngs=rngs,dtype=dtype)

    def get_up_layers(self):
        layers = []
        for i in range(self.depth):
            in_channel = self.init_channel * self.channel_multiples[i]
            out_channel = self.init_channel * self.channel_multiples[i+1]
            layers.append(Block(in_channel, out_channel, self.t_dim, rngs=self.rngs,layer=2,attn=self.attns[i],attn_head=self.attn_head,attn_dim=self.attn_headdim*self.attn_head,dtype=self.dtype,res=True))
            layers.append(AvgPool2)
        return layers

    def get_down_layers(self):
        layers = []
        for i in range(self.depth):
            in_channel = self.init_channel * self.channel_multiples[-i-1]
            out_channel = self.init_channel * self.channel_multiples[-i-2]
            # layers.append(Upsample2)
            layers.append(NeuralUpsample2(in_channel, self.rngs))
            layers.append(Block(in_channel, out_channel, self.t_dim, rngs=self.rngs,layer=2,attn=self.attns[-i-1],attn_head=self.attn_head,attn_dim=self.attn_headdim*self.attn_head,dtype=self.dtype,res=True))
        return layers

    def __call__(self,x,t):
        # show_max_and_min(x, "0")
        x = self.init_conv(x)
        # show_max_and_min(x, "1")
        # t *= 100
        t = self.t_emb(t)
        # show_max_and_min(t, "2")
        xs = []
        for i in range(len(self.up.layers)):
            if i % 2 == 0:
                x = self.up.layers[i](x,t)
                xs.append(x)
            else:
                x = self.up.layers[i](x)
            # show_max_and_min(x, f"up layer {i}")
            # print(f'up layer {i}, x.shape:', x.shape)
            # print('x stats:', x.min(), x.max(), x.mean(), x.std())
        x = self.middle(x,t)
        # show_max_and_min(x, "middle")
        # print('middle layer x.shape:', x.shape)
        # print('x stats:', x.min(), x.max(), x.mean(), x.std())
        for i in range(len(self.down.layers)):
            if i % 2 == 1:
                x = x + xs.pop()
                x = self.down.layers[i](x,t)
            else:
                x = self.down.layers[i](x)
            # show_max_and_min(x, f"down layer {i}")
            # print(f'down layer {i}, x.shape:', x.shape)
            # print('x stats:', x.min(), x.max(), x.mean(), x.std())
        x = self.end(x)
        # show_max_and_min(x, "end")
        return x

UNet_debug = partial(
    UNet,
    depth=1,
    init_channel=1,
    channel_multiples=[1,1],
    attns=[False, False],
    attn_head=1,
    attn_headdim=1,
    t_dim=2
)

UNet_for_mnist = partial( # 
    UNet,
    depth=2,
    init_channel=32,
    channel_multiples=[1,2,4],
    attns=[False,False,True],
    attn_head=4,
    attn_headdim=64,
)

UNet_for_cifar = partial(
    UNet,
    depth=3,
    init_channel=256,
    channel_multiples=[1,2,2,2],
    attns=[False,True,False,False],
    attn_head=4,
    attn_headdim=64,
)

UNet_for_32 = partial( # 27430659 params
    UNet,
    depth=3,
    init_channel=256,
    channel_multiples=[1,2,2,2],
    attns=[False,True,True,False],
    attn_head=4,
    attn_headdim=64,
)

UNet_for_64 = partial( # 47480643 params
    UNet,
    depth=3,
    init_channel=192,
    channel_multiples=[1,2,3,4],
    attns=[False,True,True,True],
    attn_head=4,
    attn_headdim=64,
)

UNet_for_128 = partial( # 83247875 params
    UNet,
    depth=4,
    init_channel=256,
    channel_multiples=[1,1,2,3,4],
    attns=[False,False,True,True,True],
    attn_head=4,
    attn_headdim=64,
)

if __name__ == '__main__':
    print('Hello world',flush=True)
    # model = UNet_debug(rngs=nn.Rngs(0))
    # model = UNet_for_mnist(rngs=nn.Rngs(0))
    model = UNet_for_32(rngs=nn.Rngs(0))
    # model = UNet_for_64(rngs=nn.Rngs(0))
    # model = UNet_for_128(rngs=nn.Rngs(0))
    # udu.show_dict(f'number of model parameters:{udu.count_params(model)}' )
    # udu.show_dict(udu.display_model(model))
    # exit()
    # x = jnp.ones((7, 32, 32, 3))
    x = jax.random.uniform(jax.random.PRNGKey(0), (7, 32, 32, 3))
    t = jax.random.uniform(jax.random.PRNGKey(0), (7,))
    y = model(x, t)
    print(y.shape)
    print('y.range:', y.min(), y.max()) # avoid NAN issue

    # # test upsample
    # x = jnp.ones((3,2,2,1))
    # # print(x)
    # print(Upsample2(x))
    # print(Upsample2(x).shape)
    # print('='*20)
    # # test downsample
    # print(AvgPool2(x))
    # print(AvgPool2(x).shape)