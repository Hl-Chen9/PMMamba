import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce, lru_cache
from typing import Optional, Callable
from operator import mul
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from selective_scan_interface import selective_scan_fn, selective_scan_ref

from models.RADDet_finetune.yolo_head import singleLayerHead, singleLayerHead_v2, singleLayerHead3D
import torch
import math


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv3d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm3d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)), 
                self.cv2(x)
            )
            , dim=1))

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x)
        x = self.fc2(x)
        return x

class PositionalEncoding3D(torch.nn.Module):
    def __init__(self, d_model, max_size=100):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        self.max_size = max_size

        pe = torch.zeros(max_size, max_size, max_size, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 6).float() * (-math.log(10000.0) / d_model))
        
        x_pos = torch.arange(0, max_size).unsqueeze(1)
        y_pos = torch.arange(0, max_size).unsqueeze(1)
        z_pos = torch.arange(0, max_size).unsqueeze(1)
        
        pe[:, :, :, 0::6] = torch.sin(x_pos * div_term).unsqueeze(1).unsqueeze(2)
        pe[:, :, :, 1::6] = torch.cos(x_pos * div_term).unsqueeze(1).unsqueeze(2)
        pe[:, :, :, 2::6] = torch.sin(y_pos * div_term).unsqueeze(0).unsqueeze(2)
        pe[:, :, :, 3::6] = torch.cos(y_pos * div_term).unsqueeze(0).unsqueeze(2)
        pe[:, :, :, 4::6] = torch.sin(z_pos * div_term).unsqueeze(0).unsqueeze(1)
        pe[:, :, :, 5::6] = torch.cos(z_pos * div_term).unsqueeze(0).unsqueeze(1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(2), :x.size(3), :x.size(4), :self.d_model].permute(3, 0, 1, 2)
        return x



def get_3d_coordinates(B, D, H, W, device, dtype=torch.float):
    """
    生成一个归一化到 [-1, 1] 的3D坐标网格。
    输出形状: (B, D, H, W, 3)
    """
    d_coords = torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype)
    h_coords = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    w_coords = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    
    # 使用 broadcasting 创建网格
    grid_d, grid_h, grid_w = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
    
    # 堆叠成 (D, H, W, 3)
    coords = torch.stack([grid_d, grid_h, grid_w], dim=-1)
    
    # 扩展到 batch 维度 -> (B, D, H, W, 3)
    return coords.unsqueeze(0).expand(B, -1, -1, -1, -1)

class SS3D(nn.Module):
    def __init__(
        self,
        d_model,   # 96
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        coord_embed_dim=64, # 坐标嵌入的中间维度
        use_coord_mod=False,
    ):
        super().__init__()
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv   # 3
        self.expand = expand   # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6
        self.use_coord_mod = use_coord_mod
        print(use_coord_mod)

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,    # 1
            bias=conv_bias,  
            groups=self.d_inner,  # 192  
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=8, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            
            
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=6, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=6, merge=True) # (K=4, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        if self.use_coord_mod:
            # param_dim = self.dt_rank + self.d_state * 2
            self.coord_net = nn.Sequential(
                nn.Linear(3, coord_embed_dim),
                nn.ReLU(),
                nn.Linear(coord_embed_dim, self.dt_rank + self.d_state)  # Δ + B
            )

        # self.num_tokens = 64
        # self.inner_rank = 128
        # self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        # self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)
        # self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        # self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        # self.route = nn.Sequential(
        #     nn.Linear(self.d_inner, self.d_inner // 4),
        #     nn.GELU(),
        #     nn.Linear(self.d_inner // 4, self.num_tokens),
        #     nn.LogSoftmax(dim=-1)
        # )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor, coords: torch.Tensor = None):
        self.selective_scan = selective_scan_fn
        
        B, C, D, H, W = x.shape
        L = D * H * W
        
        K = 6

        # 准备6个方向的输入
        # 1. D-axis scan (forward and backward)
        x_d = x.permute(0, 1, 3, 4, 2).contiguous().view(B, C, -1) # (B, C, H*W*D)
        # 2. H-axis scan (forward and backward)
        x_h = x.permute(0, 1, 2, 4, 3).contiguous().view(B, C, -1) # (B, C, D*W*H)
        # 3. W-axis scan (forward and backward)
        x_w = x.contiguous().view(B, C, -1) # (B, C, D*H*W)

        # 拼接正向和反向
        xs = torch.stack([x_d, x_h, x_w], dim=1) # (B, 3, C, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (B, K=6, C, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)

        if self.use_coord_mod and coords is not None:
            coords = coords.permute(0, 4, 1, 2, 3).contiguous()  # (B, 3, D, H, W)

            # 跟 xs 同样的 6 方向展开
            coords_d = coords.permute(0, 1, 3, 4, 2).contiguous().view(B, 3, -1)
            coords_h = coords.permute(0, 1, 2, 4, 3).contiguous().view(B, 3, -1)
            coords_w = coords.contiguous().view(B, 3, -1)

            coords_s = torch.stack([coords_d, coords_h, coords_w], dim=1)  # (B, 3, 3, L)
            coords_s = torch.cat([coords_s, torch.flip(coords_s, dims=[-1])], dim=1)  # (B, 6, 3, L)
            coords_s = coords_s.permute(0, 1, 3, 2).contiguous()  # (B, K, L, 3)

            joint_bias = self.coord_net(coords_s).permute(0, 1, 3, 2)

            # 拆分
            dt_bias = joint_bias[:, :, :self.dt_rank, :]   # (B, K, dt_rank, L)
            b_bias  = joint_bias[:, :, self.dt_rank:, :]   # (B, K, d_state, L)

            # 分别加到对应切片
            x_dbl[:, :, :self.dt_rank, :] += dt_bias
            x_dbl[:, :, self.dt_rank:self.dt_rank + self.d_state, :] += b_bias

            # b_bias = self.coord_net(coords_s).permute(0, 1, 3, 2)

            # # 只加到 B 的切片
            # b_slice = x_dbl[:, :, self.dt_rank:self.dt_rank + self.d_state, :]
            # x_dbl[:, :, self.dt_rank:self.dt_rank + self.d_state, :] = b_slice + b_bias

            # # 生成 dt 的位置偏置 → (B, K, dt_rank, L)
            # dt_bias = self.coord_net(coords_s).permute(0, 1, 3, 2)

            # # 只加到 dts 切片
            # dt_slice = x_dbl[:, :, :self.dt_rank, :]  # (B, K, dt_rank, L)
            # x_dbl[:, :, :self.dt_rank, :] = dt_slice + dt_bias
                    
        # x = x.view(B, C, L).permute(0, 2, 1)
        # full_embedding = self.embeddingB.weight @ self.embeddingA.weight  # [128, C]
        # pred_route = self.route(x)
        # cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        # prompt = torch.matmul(cls_policy, full_embedding).view(B, L, self.d_state)
        # b, l, c = prompt.shape
        # prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        y_list = self.post_scan_rearrangement_v2(out_y, B, C, D, H, W)
        return y_list

    def post_scan_rearrangement_v2(self, out_y, B, C, D, H, W):
        # out_y shape: (B, K=6, C, L)
        
        # D-axis scan outputs
        y_d_fwd = out_y[:, 0].view(B, C, H, W, D).permute(0, 1, 4, 2, 3)
        y_d_bwd = torch.flip(out_y[:, 3], dims=[-1]).view(B, C, H, W, D).permute(0, 1, 4, 2, 3)
        
        # H-axis scan outputs
        y_h_fwd = out_y[:, 1].view(B, C, D, W, H).permute(0, 1, 2, 4, 3)
        y_h_bwd = torch.flip(out_y[:, 4], dims=[-1]).view(B, C, D, W, H).permute(0, 1, 2, 4, 3)

        # W-axis scan outputs
        y_w_fwd = out_y[:, 2].view(B, C, D, H, W)
        y_w_bwd = torch.flip(out_y[:, 5], dims=[-1]).view(B, C, D, H, W)

        return [
            y_d_fwd.contiguous().view(B, -1, D*H*W),
            y_d_bwd.contiguous().view(B, -1, D*H*W),
            y_h_fwd.contiguous().view(B, -1, D*H*W),
            y_h_bwd.contiguous().view(B, -1, D*H*W),
            y_w_fwd.contiguous().view(B, -1, D*H*W),
            y_w_bwd.contiguous().view(B, -1, D*H*W),
        ]
    
    def forward(self, x: torch.Tensor):
        B, D, H, W, C = x.shape
        coords = None
        if self.use_coord_mod:
            coords = get_3d_coordinates(B, D, H, W, device=x.device, dtype=x.dtype)

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, d, h, w, c)  # x走的是ss2d的路径

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x)) # (b, c, d, h, w) 
        y_list = self.forward_core(x, coords)
        y = sum(y_list)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, D, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,  # 96
        drop_path: float = 0,  # 0.2
        attn_drop_rate: float = 0,  # 0
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
        d_state: int = 16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.self_attention = SS3D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.ffn = ConvFFN(in_features=hidden_dim, hidden_features=hidden_dim*2, kernel_size=3)
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 4, 1)
        x = x + self.drop_path1(self.self_attention(self.ln_1(x)))
        x = x + self.drop_path2(self.ffn(self.ln_2(x)))
        x = x.permute(0, 4, 1, 2, 3)
        return x

class Stem(nn.Module):
    def __init__(self, in_chans,embed_dim):
        super().__init__()
        self.in_chans = in_chans

        self.silu = nn.SiLU()

        self.conv1a = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv1b = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.gn1a = nn.GroupNorm(num_groups=int(embed_dim/4), num_channels=embed_dim)
        self.gn1b = nn.GroupNorm(num_groups=int(embed_dim/4), num_channels=embed_dim)

    def forward(self, x):

        x = self.silu(self.gn1a(self.conv1a(x)))
        x = self.silu(self.gn1b(self.conv1b(x)))

        return x
    
class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, in_chans,embed_dim):
        super().__init__()

        self.dim = embed_dim
        self.silu = nn.SiLU()
        self.conv2a = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.gn2a  = nn.GroupNorm(num_groups=int(embed_dim/4), num_channels=embed_dim)

    def forward(self, x):    
        x = self.silu(self.gn2a(self.conv2a(x)))
        return x

    
class MambaBackbone(nn.Module):
    def __init__(self, depth=[2, 2, 2], in_chans=2, embed_dim=[64, 128, 320],
                attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6) 
        self.stem = Stem(in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchMerging(in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchMerging(in_chans=embed_dim[1], embed_dim=embed_dim[2])

        self.PE3D = PositionalEncoding3D(embed_dim[0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList([
            VSSBlock(hidden_dim=embed_dim[0], drop_path=dpr[i], attn_drop_rate=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            VSSBlock(hidden_dim=embed_dim[1], drop_path=dpr[i + depth[0]], attn_drop_rate=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            VSSBlock(hidden_dim=embed_dim[2], drop_path=dpr[i + depth[0] + depth[1]], attn_drop_rate=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth[2])])

    
    def forward_features(self, x):
        features = []
        x = self.stem(x)
        x = self.PE3D(x)
        for blk in self.blocks1:
            x = blk(x)
        features.append(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        features.append(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        features.append(x)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        return features

    
class RadarMamba_fpn(nn.Module):
    def __init__(self, depth=[2, 2, 2], embed_dim=[48, 96, 192]):
        super(RadarMamba_fpn, self).__init__()
        
        self.backbone = MambaBackbone(depth=depth, embed_dim=embed_dim)
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(embed_dim[2], embed_dim[1], 1, 1)
        self.conv3_for_upsample1    = C3(embed_dim[2], embed_dim[1], 2, shortcut=False)

        self.conv_for_feat2         = Conv(embed_dim[1], embed_dim[0], 1, 1)
        self.conv3_for_upsample2    = C3(embed_dim[1], embed_dim[0], 2, shortcut=False)

        self.down_sample1           = Conv(embed_dim[0], embed_dim[0],3, 2, 1)
        self.conv3_for_downsample1  = C3(embed_dim[1], embed_dim[1], 2, shortcut=False)

        self.down_sample2           = Conv(embed_dim[1], embed_dim[1], 3, 2)
        self.conv3_for_downsample2 = C3(embed_dim[2], embed_dim[2], 2, shortcut=False)
        self.yolo_head_P3 = singleLayerHead3D(num_anchors=6,
                                         num_class=6,
                                         last_channel=0,
                                         channel=embed_dim[0])
        
        self.yolo_head_P4 = singleLayerHead3D(num_anchors=6,
                                         num_class=6,
                                         last_channel=0,
                                         channel=embed_dim[1])
        
        self.yolo_head_P5 = singleLayerHead3D(num_anchors=6,
                                         num_class=6,
                                         last_channel=0,
                                         channel=embed_dim[2])

    def forward(self, imgs):
        features = self.backbone(imgs)
        feat1, feat2, feat3 = features[0], features[1], features[2] # Shapes: (48, 16, 64, 64), (96, 8, 32, 32), (192, 4, 16, 16)

        # --- Top-down FPN path ---
        p5_from_backbone = self.conv_for_feat3(feat3) # (192->96 channels), shape: (B, 96, 4, 16, 16)
        
        p5_upsampled = self.upsample(p5_from_backbone)
        p4_topdown = torch.cat([p5_upsampled, feat2], 1) # (96+96=192 channels)
        p4_topdown = self.conv3_for_upsample1(p4_topdown) # (192->96 channels), shape: (B, 96, 8, 32, 32)

        p4_for_p3_fusion = self.conv_for_feat2(p4_topdown) # (96->48 channels)
        p4_upsampled = self.upsample(p4_for_p3_fusion)
        p3_out = torch.cat([p4_upsampled, feat1], 1) # (48+48=96 channels)
        p3_out = self.conv3_for_upsample2(p3_out) # (96->48 channels), shape: (B, 48, 16, 64, 64)
        
        # --- Bottom-up PANet path ---
        p3_downsampled = self.down_sample1(p3_out) # (48->48 channels, s=2)
        p4_out = torch.cat([p3_downsampled, p4_for_p3_fusion], 1) # (48+48=96 channels)
        p4_out = self.conv3_for_downsample1(p4_out) # (96->96 channels), shape: (B, 96, 8, 32, 32)
        
        p4_downsampled = self.down_sample2(p4_out) # (96->96 channels, s=2)
        p5_out = torch.cat([p4_downsampled, p5_from_backbone], 1) # (96+96=192 channels)
        p5_out = self.conv3_for_downsample2(p5_out) # (192->192 channels), shape: (B, 192, 4, 16, 16)
        
        # --- YOLO Heads ---
        out2 = self.yolo_head_P3(p3_out)
        out1 = self.yolo_head_P4(p4_out)
        out0 = self.yolo_head_P5(p5_out)
        
        return out1, out0, out2

def Mamba_raddet(**kwargs):
    model = RadarMamba_fpn(
        depth=[2, 2, 2], embed_dim=[48, 96, 192]
    )
    return model


from fvcore.nn import FlopCountAnalysis, parameter_count_table    
if __name__ == '__main__':
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    input = torch.rand(1, 2, 64, 256, 256).to(device)
    net = RadarMamba_fpn(depth=[2, 2, 2], embed_dim=[48, 96, 192]).to(device)
    out0, out1, out2 = net(input)
    print(out0.shape)
    print(out1.shape)
    print(out2.shape)

    dummy_input = torch.randn(1, 2, 64, 256, 256).to(device)
    flop_counter = FlopCountAnalysis(net, dummy_input)
    print(flop_counter)
    gflops = flop_counter.total() / 1e9  # 将flops转换为gflops
    print(f"GFLOPS: {gflops}")
    total_params = sum(p.numel() for p in net.parameters())

    total_params_millions = total_params / 1e6
    print(f"Total Parameters: {total_params_millions:.2f}M")
