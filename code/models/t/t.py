import sys
sys.path.append('/kmh-nfs-ssd-eu-mount/code/qiao/work/sqa-flow-matching/code/models')

import jax
import jax.numpy as jnp
import flax.nnx as nn
from jcm import layers, layerspp, normalization
from functools import partial
import numpy as np

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

    def get_beta_schedule(self):
        beta_schedule = 'linear'
        beta_start = 1e-4
        beta_end = 0.02
        num_diffusion_timesteps = 1000
        def sigmoid(x):
            return 1 / (jnp.exp(-x) + 1)

        if beta_schedule == "quad":
            betas = (
                jnp.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
            )
        elif beta_schedule == "linear":
            betas = jnp.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "const":
            betas = beta_end * jnp.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / jnp.linspace(
                num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "sigmoid":
            betas = jnp.linspace(-6, 6, num_diffusion_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_diffusion_timesteps,)
        return betas

    def get_nearest_index(self, t):
        t = 1-t
        betas = self.get_beta_schedule()
        alpha = jnp.cumprod(1-betas, axis=0) # (1000, )
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