import torch.nn as nn
from .mlp import DropPath, GELU
from .conv import SparseDepthwiseConv2d
from .norm import SpLayerNorm, SpGRN

class ConvBlock(nn.Module):
    """ Sparse ConvNeXtV2 Block. 

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0.):
        super(ConvBlock, self).__init__()
        self.dwconv = SparseDepthwiseConv2d(dim, kernel_size=7)
        self.norm = SpLayerNorm(dim, 1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)   
        self.act = GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.grn = SpGRN(4  * dim)
        self.drop_path = DropPath(drop_path)

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