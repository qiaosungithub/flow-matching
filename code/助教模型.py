import jax
import jax.numpy as jnp
import flax.nnx as nn
import zhh.F as F
from functools import partial
from utils.utils import get_sigmas
# from config import CONFIG

class GELU(nn.Module):
    def __call__(self,x):
        return 0.5 * x * (1 + jnp.tanh(0.7978845608 * (x + 0.044715 * x**3)))

class FCEmb(nn.Module):

    def __init__(self,dim, rngs):
        self.net = nn.Sequential(
            nn.Linear(1, dim, rngs=rngs),
            GELU(),
            nn.Linear(dim, dim, rngs=rngs),
        )

    def __call__(self,x):
        return self.net(x.unsqueeze(-1))
    
# class Attention(nn.Module):

#     def __init__(self,dim,head,rngs,attn_dim=None):
#         self.head = head
#         self.scale = dim ** -0.5
#         self.attn_dim = attn_dim if attn_dim is not None else dim//2
#         self.head_dim = self.attn_dim // head
#         self.Q = nn.Conv(dim, self.attn_dim,kernel_size=(1, 1), use_bias=False, rngs=rngs)
#         self.K = nn.Conv(dim, self.attn_dim,kernel_size=(1, 1), use_bias=False, rngs=rngs)
#         self.V = nn.Conv(dim, self.attn_dim,kernel_size=(1, 1), use_bias=False, rngs=rngs)
#         self.out = nn.Conv(self.attn_dim, dim,kernel_size=(1, 1), use_bias=False, rngs=rngs)

#     def __call__(self,query, context):
#         b,c,h,w = query.shape
#         q = self.Q(query).reshape(b, self.head, self.head_dim, h*w)
#         k = self.K(context).reshape(b, self.head, self.head_dim, h*w)
#         v = self.V(context).reshape(b, self.head, self.head_dim, h*w)
#         score = jnp.einsum('bhdi,bhdj->bhij', q, k) * self.scale
#         attn = nn.softmax(score, dim=-1)
#         out = jnp.einsum('bhij,bhdj->bhdi', attn, v).reshape(b, self.attn_dim, h, w)
#         return self.out(out)
    
class ScaleOnlyLayerNorm(nn.Module):

    def __init__(self, dim, rngs):
        # self.scale = nn.Parameter(jnp.ones(dim))
        self.scale = nn.Embed(num_embeddings=1, features=dim, embedding_init=nn.initializers.ones, rngs=rngs)
        self.norm = nn.LayerNorm(dim,use_bias=False, use_scale=False, rngs=rngs)
    
    def __call__(self, x):
        # x = x.permute(0,2,3,1)
        x = jnp.transpose(x, (0,2,3,1))
        # return (self.norm(x) * self.scale).permute(0,3,1,2)
        return jnp.transpose(self.norm(x) * self.scale, (0,3,1,2))
    
class ResBlock(nn.Module):

    def __init__(self,in_channel, out_channel, rngs, kernel=(3, 3)):
        self.in_channel = in_channel
        self.out_channel = out_channel
        if type(kernel) == int:
            kernel = (kernel, kernel)
        padding = (kernel[0]//2, kernel[1]//2)
        self.conv1 = nn.Sequential(
            nn.Conv(in_channel,out_channel,kernel,padding=padding, rngs=rngs),
            nn.BatchNorm(out_channel, rngs=rngs),
            GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv(out_channel,out_channel,kernel,padding=padding, rngs=rngs),
            nn.BatchNorm(out_channel, rngs=rngs),
            GELU(),
        )

    def __call__(self,x):
        x = self.conv1(x)
        xc = x
        x = self.conv2(x)
        return x + xc

class Block(nn.Module):

    def __init__(self,in_channels,out_channels,t_dim,rngs,layer=2,kernel=3,attn=False,attn_head=4,attn_dim=None,res=False):
        self.have_attn = attn
        # self.init_conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel,padding=kernel//2)
        # self.t_net = nn.Linear(t_dim,out_channels)
        self.t_net = FCEmb(in_channels, rngs=rngs)
        self.layers = layer
        if attn:
            raise NotImplementedError()
            self.attn = nn.ModuleList([Attention(out_channels,attn_head,attn_dim) for _ in range(layer)])
            self.pre_norm = ScaleOnlyLayerNorm(out_channels)
        self.have_res = res
        if res:
            self.res = [ResBlock(in_channel=out_channels,out_channel=out_channels, rngs=rngs) for _ in range(layer)]

    def __call__(self,x,t):
        t1 = self.t_net(t)
        x = x + t1.unsqueeze(-1).unsqueeze(-1)
        for l in range(self.layers):
            if self.have_attn:
                raise NotImplementedError()
                x = self.pre_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
                x = self.attn[l](x,x)
            if self.have_res:
                x = self.res[l](x)
        return x
    
class Sigmas(nn.Variable): pass

class UNet(nn.Module):

    def __init__(self, 
        ngf=32,
        rngs=None, 
        n_noise_levels: int=None,
        config: dict={},
        dtype: jnp.dtype = jnp.float32,
    ):
        self.config = config
        self.dtype = dtype
        self.init_conv = nn.Conv(1,128,kernel_size=(3, 3),padding=(1, 1), rngs=rngs)
        # assert CONFIG.condition_type == 't', NotImplementedError()

        # self.t_emb = nn.Embedding(CONFIG.L,t_dim)
        self.t_emb = FCEmb(ngf, rngs=rngs)
        self.up = [
            # Block(32,64,t_dim,layer=2,res=True,attn=False),
            ResBlock(128,128, rngs),
            partial(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            # Block(64,128,t_dim,layer=2,res=True,attn=False),
            ResBlock(128,256, rngs),
            partial(nn.avg_pool, window_shape=(2, 2), strides=(2, 2)),
        ]
        # self.middle = Block(128,128,t_dim,layer=2,res=True,attn=True)
        self.middle = nn.Sequential(
            # ResBlock(256,256),
            # ResBlock(256,256),
            partial(nn.avg_pool, window_shape=(7, 7), strides=(7, 7)),
            GELU(),
            nn.ConvTranspose(in_features=256,out_features=256,kernel_size=(7,7),strides=(7, 7), padding=(0, 0), rngs=rngs),
        )
        self.down = [
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose(in_features=512,out_features=128,kernel_size=(2,2),strides=(2, 2), padding=(0, 0), rngs=rngs),
            Block(128,128,t_dim=1,layer=2,res=True,attn=False, rngs=rngs),
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose(in_features=256,out_features=128,kernel_size=(2,2),strides=(2, 2), padding=(0, 0), rngs=rngs),
            Block(128,128,t_dim=1,layer=2,res=True,attn=False, rngs=rngs),
        ]
        # self.end = nn.Conv2d(32,1,kernel_size=3,padding=1)
        self.end = nn.Sequential(
            nn.Conv(256,128,kernel_size=(3, 3),padding=(1, 1), rngs=rngs),
            ScaleOnlyLayerNorm(128, rngs),
            GELU(),
            nn.Conv(128,1,kernel_size=(3, 3),padding=(1, 1), rngs=rngs),
        )

        self.sigmas = get_sigmas(self.config.sampling)
        self.sigmas = self.sigmas.astype(self.dtype)
        self.sigmas = Sigmas(self.sigmas)

    def __call__(self,x,t):
        x = self.init_conv(x)
        xc = x
        # t = t.float()
        t = self.sigmas[t]
        # t = self.t_emb(t.float())
        xs = []
        for ly in self.up:
            if isinstance(ly,ResBlock):
                x = ly(x)
            else: # AvgPool2d
                x = ly(x)
                xs.append(x)
        # x = self.middle(x,t)
        x = self.middle(x)
        # print('after middle, x.shape:', x.shape)
        # print('cached shape:',[c.shape for c in xs])
        for ly in self.down:
            if isinstance(ly,nn.ConvTranspose):
                # x = x + xs.pop()
                x = jnp.concatenate([x,xs.pop()],axis=-1) # here the channel is the last axis
                x = ly(x)
            else: # Upsample
                x = ly(x,t)
        return self.end(jnp.concatenate([x,xc],axis=-1))
    
if __name__ == '__main__':
    model = UNet()
    # print('number of parameters:', sum(p.numel() for p in model.parameters()))
    # print(model(torch.randn(7,1,28,28),torch.randn(7,)).shape)
    print(model)