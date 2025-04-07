import torch
import torch.nn as nn
from einops import rearrange

def conv_1x1_bn(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(ch_out),
        nn.SiLU()
    )
def conv_nxn_bn(ch_in, ch_out, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(ch_out),
        nn.SiLU()
    )
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class TAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)
class MoblieTrans(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TAttention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
class MV2Block(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(ch_in * expansion)
        self.use_res_connect = self.stride == 1 and ch_in == ch_out
        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch_out),
            )
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
class MobileViTBlock(nn.Module):
    def __init__(self, ch_in, dim=64, depth=2, kernel_size=3, patch_size=(2, 2), mlp_dim=int(64*2), dropout=0.):
        super().__init__() # mg
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(ch_in, ch_in, kernel_size)
        self.conv2 = conv_1x1_bn(ch_in, dim)
        self.transformer = MoblieTrans(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, ch_in)
        self.conv4 = conv_nxn_bn(2 * ch_in, ch_in, kernel_size)
    
    def forward(self, x):
        y = x.clone()# torch.Size for 
        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)# torch.Size for 
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x) 
        return x

