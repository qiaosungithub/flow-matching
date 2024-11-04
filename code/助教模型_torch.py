import torch
import torch.nn as nn
import torch.nn.functional as F
# from config import CONFIG

class GELU(nn.Module):
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))

class FCEmb(nn.Module):

    def __init__(self,dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self,x):
        return self.net(x.unsqueeze(-1))
    
class Attention(nn.Module):

    def __init__(self,dim,head,attn_dim=None):
        super().__init__()
        self.head = head
        self.scale = dim ** -0.5
        self.attn_dim = attn_dim if attn_dim is not None else dim//2
        self.head_dim = self.attn_dim // head
        self.Q = nn.Conv2d(dim, self.attn_dim,kernel_size=1, bias=False)
        self.K = nn.Conv2d(dim, self.attn_dim,kernel_size=1, bias=False)
        self.V = nn.Conv2d(dim, self.attn_dim,kernel_size=1, bias=False)
        self.out = nn.Conv2d(self.attn_dim, dim,kernel_size=1, bias=False)

    def forward(self,query, context):
        b,c,h,w = query.shape
        q = self.Q(query).reshape(b, self.head, self.head_dim, h*w)
        k = self.K(context).reshape(b, self.head, self.head_dim, h*w)
        v = self.V(context).reshape(b, self.head, self.head_dim, h*w)
        score = torch.einsum('bhdi,bhdj->bhij', q, k) * self.scale
        attn = F.softmax(score, dim=-1)
        out = torch.einsum('bhij,bhdj->bhdi', attn, v).reshape(b, self.attn_dim, h, w)
        return self.out(out)
    
class ScaleOnlyLayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.norm = nn.LayerNorm(dim,elementwise_affine=False)
    
    def forward(self, x):
        x = x.permute(0,2,3,1)
        return (self.norm(x) * self.scale).permute(0,3,1,2)
    
class ResBlock(nn.Module):

    def __init__(self,in_channel, out_channel,kernel=3):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel,padding=kernel//2),
            nn.BatchNorm2d(out_channel),
            GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel,padding=kernel//2),
            nn.BatchNorm2d(out_channel),
            GELU(),
        )

    def forward(self,x):
        x = self.conv1(x)
        xc = x.clone()
        x = self.conv2(x)
        return x + xc

class Block(nn.Module):

    def __init__(self,in_channels,out_channels,t_dim,layer=2,kernel=3,attn=False,attn_head=4,attn_dim=None,res=False):
        super().__init__()
        self.have_attn = attn
        # self.init_conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel,padding=kernel//2)
        # self.t_net = nn.Linear(t_dim,out_channels)
        self.t_net = FCEmb(in_channels)
        self.layers = layer
        if attn:
            raise NotImplementedError()
            self.attn = nn.ModuleList([Attention(out_channels,attn_head,attn_dim) for _ in range(layer)])
            self.pre_norm = ScaleOnlyLayerNorm(out_channels)
        self.have_res = res
        if res:
            self.res = nn.ModuleList([ResBlock(in_channel=out_channels,out_channel=out_channels) for _ in range(layer)])

    def forward(self,x,t):
        t1 = self.t_net(t)
        x = x + t1.unsqueeze(-1).unsqueeze(-1)
        for l in range(self.layers):
            if self.have_attn:
                x = self.pre_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
                x = self.attn[l](x,x)
            if self.have_res:
                x = self.res[l](x)
        return x
    
class UNet(nn.Module):

    def __init__(self, t_dim=32):
        super().__init__()
        self.init_conv = nn.Conv2d(1,128,kernel_size=3,padding=1)
        assert CONFIG.condition_type == 't', NotImplementedError()

        # self.t_emb = nn.Embedding(CONFIG.L,t_dim)
        self.t_emb = FCEmb(t_dim)
        self.up = nn.Sequential(
            # Block(32,64,t_dim,layer=2,res=True,attn=False),
            ResBlock(128,128),
            nn.AvgPool2d(2),
            # Block(64,128,t_dim,layer=2,res=True,attn=False),
            ResBlock(128,256),
            nn.AvgPool2d(2),
        )
        # self.middle = Block(128,128,t_dim,layer=2,res=True,attn=True)
        self.middle = nn.Sequential(
            # ResBlock(256,256),
            # ResBlock(256,256),
            nn.AvgPool2d(7),
            GELU(),
            nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=7,stride=7),
        )
        self.down = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=512,out_channels=128,kernel_size=2,stride=2),
            Block(128,128,t_dim=1,layer=2,res=True,attn=False),
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2),
            Block(128,128,t_dim=1,layer=2,res=True,attn=False),
        )
        # self.end = nn.Conv2d(32,1,kernel_size=3,padding=1)
        self.end = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,padding=1),
            ScaleOnlyLayerNorm(128),
            GELU(),
            nn.Conv2d(128,1,kernel_size=3,padding=1),
        )

    def forward(self,x,t):
        x = self.init_conv(x)
        xc = x.clone()
        t = t.float()
        # t = self.t_emb(t.float())
        xs = []
        for i,ly in enumerate(self.up):
            if isinstance(ly,ResBlock):
                x = ly(x)
            else: # AvgPool2d
                x = ly(x)
                xs.append(x)
        # x = self.middle(x,t)
        x = self.middle(x)
        # print('after middle, x.shape:', x.shape)
        # print('cached shape:',[c.shape for c in xs])
        for i,ly in enumerate(self.down):
            if isinstance(ly,nn.ConvTranspose2d):
                # x = x + xs.pop()
                x = torch.cat([x,xs.pop()],dim=1)
                x = ly(x)
            else: # Upsample
                x = ly(x,t)
        return self.end(torch.cat([x,xc],dim=1))
    
if __name__ == '__main__':
    model = UNet()
    print('number of parameters:', sum(p.numel() for p in model.parameters()))
    print(model(torch.randn(7,1,28,28),torch.randn(7,)).shape)