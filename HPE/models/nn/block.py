import torch
import torch.nn as nn
from torch import Tensor
from .mlp import MLP, SpDropPath, SparseGELU
from .norm import SpLayerNorm, SpGRN, LayerNorm
from .conv import SparseDepthwiseConv2d
import spconv.pytorch as spconv

class SpBlock(nn.Module):
    """ Sparse ConvNeXtV2 Block. 

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0.):
        super(SpBlock, self).__init__()
        self.dwconv = SparseDepthwiseConv2d(dim, kernel_size=7)
        self.norm = SpLayerNorm(dim, 1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)   
        self.act = SparseGELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.grn = SpGRN(4  * dim)
        self.drop_path = SpDropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., use_gated=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if use_gated:
            from .attention import GatedAttention
            self.attn = GatedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                      attn_drop=attn_drop, proj_drop=drop)
        else:
            from .attention import Attention
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                 attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ConvNext2Block(nn.Module):
    """ConvNeXt V2 Block for 2D data"""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_init_value = layer_scale_init_value
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x