import sys
sys.path.append('/kmh-nfs-ssd-eu-mount/code/qiao/work/sqa-flow-matching/code/models')

import jax.numpy as jnp
import flax.nnx as nn
from jcm import layers, layerspp, normalization
from functools import partial

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
                 **kwargs):
        # for restoring checkpoints #
        base_width = 20
        act = "relu"
        use_sigmoid = False
        ##############################

        self.conv1 = conv3x3(3, base_width, rngs=rngs)
        self.conv2 = conv3x3(base_width, base_width, rngs=rngs)
        self.act = get_act(act)
        self.pool = nn.avg_pool
        self.fc = nn.Linear(base_width, 1, rngs=rngs)
        self.use_sigmoid = use_sigmoid
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x, (32, 32))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        if self.use_sigmoid:
            x = nn.sigmoid(x)
        return x

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