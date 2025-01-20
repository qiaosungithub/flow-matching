import sys
sys.path.append('/kmh-nfs-ssd-eu-mount/code/qiao/work/sqa-flow-matching/code/models')

import jax
import jax.numpy as jnp
import flax.nnx as nn
from jcm import layers, layerspp, normalization
from functools import partial
import numpy as np
import math

conv3x3 = layerspp.conv3x3
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp

# class sqa_t_toy(nn.Module):

#     def __init__(self, 
#                  base_width = 10,
#                  act = "relu",
#                  dtype = jnp.float32, 
#                  use_sigmoid = True,
#                  rngs=None, 
#                  **kwargs):
#         self.fc1 = nn.Linear(32*32*3, base_width, rngs=rngs)
#         self.fc2 = nn.Linear(base_width, 1, rngs=rngs)
#         self.act = get_act(act)
#         self.use_sigmoid = use_sigmoid
    
#     def forward(self, x):
#         x = jnp.reshape(x, (x.shape[0], -1))
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         # if self.use_sigmoid:
#         #     x = nn.sigmoid(x)
#         # x = (jnp.mean(x**2, axis=(1, 2, 3)))**(0.5)
#         return x

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return jnp.array(betas)

def create_zhh_diffusion_schedule():
    if True:
        betas = betas_for_alpha_bar(
                500,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        # 大便
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.append(1.0, alphas_cumprod[:-1])
        alphas_cumprod_next = jnp.append(alphas_cumprod[1:], 0.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = jnp.log(jnp.append(posterior_variance[1], posterior_variance[1:]))
        zhh_diffusion_schedule = {
        'betas': betas, 'alphas': alphas, 'alphas_cumprod': alphas_cumprod, 'alphas_cumprod_prev': alphas_cumprod_prev, 'alphas_cumprod_next': alphas_cumprod_next,
        'posterior_variance': posterior_variance, 'posterior_log_variance_clipped': posterior_log_variance_clipped,
        }
        # cosine schedule
    elif False:
        raise NotImplementedError
    else:
        raise NotImplementedError
    return zhh_diffusion_schedule

class sqa_t_ver1(nn.Module):

    def __init__(self, 
                 base_width = 10,
                 act = "relu",
                 dtype = jnp.float32, 
                 use_sigmoid = True,
                 rngs=None, 
                 round=True,
                 **kwargs):
        # for restoring checkpoints #
        base_width = 20 # VP model
        act = "relu"
        use_sigmoid = False
        ##############################

        self.conv1 = conv3x3(3, base_width, rngs=rngs)
        self.conv2 = conv3x3(base_width, base_width, rngs=rngs)
        self.act = get_act(act)
        self.pool = nn.avg_pool
        self.fc = nn.Linear(base_width, 1, rngs=rngs)
        self.use_sigmoid = use_sigmoid
        self.round = round # whether to round the output to the nearest integer
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x, (32, 32))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        if self.use_sigmoid:
            x = nn.sigmoid(x) # x shape: (bs, 1)
        if self.round:
            # x = x.squeeze(-1)
            x = self.get_nearest_index(x)
            x = x.astype(jnp.float32)
            # assert x.ndim==1
            x = x.reshape(x.shape[0], 1)
        return x

    def get_nearest_index(self, t):
        t = 1-t
        alpha = create_zhh_diffusion_schedule()['alphas_cumprod']
        dif = jnp.abs(alpha - t)
        # assert dif.shape[1] == 1000
        index = jnp.argmin(dif, axis=1) 
        # assert index.shape == (t.shape[0],), f"get shape {index.shape}"
        # print(index, flush=True)
        # jax.debug.print('index: {s}', s=index.shape)
        # index = 999 - index
        return index

# class sqa_t_ver2(nn.Module):

#     def __init__(self, 
#                  base_width = 10,
#                  act = "relu",
#                  dtype = jnp.float32, 
#                  use_sigmoid = True,
#                  rngs=None, 
#                  **kwargs):
#         self.conv1 = nn.Conv(3, base_width, kernel_size=(5, 5), strides=(2,2), padding="SAME", rngs=rngs)
#         self.conv2 = nn.Conv(base_width, base_width, kernel_size=(5, 5), strides=(2,2), padding="SAME", rngs=rngs)
#         self.conv3 = nn.Conv(base_width, base_width, kernel_size=(5, 5), strides=(2,2), padding="SAME", rngs=rngs)
#         self.act = get_act(act)
#         self.pool = nn.avg_pool
#         self.fc1 = nn.Linear(16 * base_width, base_width, rngs=rngs)
#         self.fc2 = nn.Linear(base_width, 1, rngs=rngs)
#         self.use_sigmoid = use_sigmoid
    
#     def forward(self, x):
#         x = self.conv1(x)
#         assert x.shape[2] == 16
#         x = self.act(x)
#         x = self.conv2(x)
#         x = self.act(x)
#         x = self.conv3(x)
#         x = self.act(x)
#         x = jnp.reshape(x, (x.shape[0], -1))
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         if self.use_sigmoid:
#             x = nn.sigmoid(x)
#         return x

# class sqa_t_ver3(nn.Module):

#     def __init__(self, 
#                  base_width = 10,
#                  act = "relu",
#                  dtype = jnp.float32, 
#                  use_sigmoid = True,
#                  rngs=None, 
#                  **kwargs):
#         self.conv1 = conv3x3(3, base_width, rngs=rngs)
#         self.act = get_act(act)
#         ResnetBlock = partial(
#             ResnetBlockBigGAN,
#             act=self.act,
#             dropout=0,
#             fir=True,
#             fir_kernel=(1, 3, 3, 1),
#             init_scale=0.0,
#             skip_rescale=True,
#             rngs=rngs,
#         )
#         self.resnet = nn.Sequential(
#             ResnetBlock(base_width, out_ch=base_width),
#             ResnetBlock(base_width, out_ch=base_width),
#             ResnetBlock(base_width, out_ch=base_width*2, down=True),
#             ResnetBlock(base_width*2, out_ch=base_width*2),
#             ResnetBlock(base_width*2, out_ch=base_width*2),
#             ResnetBlock(base_width*2, out_ch=base_width*4, down=True),
#             ResnetBlock(base_width*4, out_ch=base_width*4),
#             ResnetBlock(base_width*4, out_ch=base_width*4),
#             ResnetBlock(base_width*4, out_ch=base_width*8, down=True),
#         )
        
#         # self.pool = nn.avg_pool
#         self.fc1 = nn.Linear(16 * 8 * base_width, base_width, rngs=rngs)
#         self.fc2 = nn.Linear(base_width, 1, rngs=rngs)
#         self.use_sigmoid = use_sigmoid
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.act(x)
#         x = self.resnet(x)
#         assert x.shape[2] == 4
#         x = jnp.reshape(x, (x.shape[0], -1))
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         if self.use_sigmoid:
#             x = nn.sigmoid(x)
#         return x

# class sqa_t_ver4(nn.Module):
#     """
#     This is a deep network
#     """

#     def __init__(self, 
#                  base_width = 10,
#                  act = "relu",
#                  dtype = jnp.float32, 
#                  use_sigmoid = True,
#                  rngs=None, 
#                  **kwargs):
#         self.act = get_act(act)
#         self.conv1 = conv3x3(3, base_width, rngs=rngs)
#         self.conv2 = conv3x3(base_width, base_width*2, rngs=rngs)
#         # self.pool1 = nn.AvgPool2d(2, 2)
#         self.conv3 = conv3x3(base_width*2, base_width*4, rngs=rngs)
#         # self.conv4 = conv3x3(64, 64)
#         self.in_c = 64
#         ResnetBlock = partial(
#             ResnetBlockBigGAN,
#             act=self.act,
#             dropout=0,
#             fir=True,
#             fir_kernel=(1, 3, 3, 1),
#             init_scale=0.0,
#             skip_rescale=True,
#             rngs=rngs,
#         )
#         self.res_layer = nn.Sequential(
#             ResnetBlock(base_width*4, out_ch=base_width*4),
#             ResnetBlock(base_width*4, out_ch=base_width*4),
#             ResnetBlock(base_width*4, out_ch=base_width*4),
#         )
#         self.fc1 = nn.Linear(base_width * 4 * 16 * 16, base_width * 16, rngs=rngs)
#         self.fc2 = nn.Linear(base_width * 16, 1, rngs=rngs)
#         self.use_sigmoid = use_sigmoid
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.act(x)
#         x = self.conv2(x)
#         x = self.act(x)
#         x = nn.avg_pool(x, (2, 2), strides=(2, 2))
#         x = self.conv3(x)
#         x = self.act(x)
#         x = self.res_layer(x)
#         assert x.shape[2] == 16
#         # print(x.shape)
#         x = jnp.reshape(x, (x.shape[0], -1))
#         # print(x.shape)
#         # exit("邓东灵")
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
        
#         if self.use_sigmoid:
#             x = nn.sigmoid(x)
#         return x

def get_act(s):
    """
    config: the model config
    """
    if s == 'elu':
        return nn.elu
    elif s == 'relu':
        return nn.relu
    elif s == 'lrelu':
        return partial(nn.leaky_relu, negative_slope=0.2)
    elif s == 'swish':
        def swish(x):
            return x * nn.sigmoid(x)
        return swish
    else:
        raise NotImplementedError('activation function does not exist!')