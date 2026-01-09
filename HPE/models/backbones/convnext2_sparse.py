# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import spconv.pytorch as spconv
from timm.models.layers import trunc_normal_
from nn.spmodules import ConvBlock, LayerNorm, LayerNorm

def to_sparse(x):
    """
    Dense tensor를 sparse tensor로 변환
    
    Args:
        x: Dense tensor [B, C, H, W]
    
    Returns:
        spconv.SparseConvTensor
    """
    batch_size, in_channels, H, W = x.shape
    
    # Non-zero 위치 찾기 (모든 채널에서 0이 아닌 위치)
    # 각 spatial location에서 하나라도 0이 아닌 값이 있으면 유지
    mask = (x.abs().sum(dim=1) > 1e-6)  # [B, H, W]
    
    indices_list = []
    features_list = []
    
    for b in range(batch_size):
        # 현재 batch의 non-zero 위치
        nz_indices = torch.nonzero(mask[b], as_tuple=False)  # [N, 2] (y, x)
        
        if len(nz_indices) > 0:
            # batch index 추가: [N, 3] (batch_idx, y, x)
            batch_indices = torch.full((len(nz_indices), 1), b, dtype=torch.int32, device=x.device)
            batch_nz_indices = torch.cat([batch_indices, nz_indices.int()], dim=1)
            indices_list.append(batch_nz_indices)
            
            # 해당 위치의 features 추출
            y_coords = nz_indices[:, 0]
            x_coords = nz_indices[:, 1]
            feats = x[b, :, y_coords, x_coords].T  # [N, C]
            features_list.append(feats)
    
    # 모든 batch 결합
    if len(indices_list) > 0:
        indices = torch.cat(indices_list, dim=0)
        features = torch.cat(features_list, dim=0)
    else:
        # 모든 픽셀이 0인 경우 처리
        indices = torch.zeros((0, 3), dtype=torch.int32, device=x.device)
        features = torch.zeros((0, in_channels), dtype=x.dtype, device=x.device)
    
    # Sparse tensor 생성
    sparse_tensor = spconv.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=[H, W],
        batch_size=batch_size
    )
    
    return sparse_tensor


class SparseConvNeXtV2(nn.Module):
    """ Sparse ConvNeXtV2.
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        D (int): Dimension for spconv (2 for 2D, 3 for 3D). Default: 2
    """
    def __init__(self, 
                 in_chans=3, 
                 num_classes=1000, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 D=2):  # 2D 이미지의 경우 D=2
        super().__init__()
        self.depths = depths
        self.num_classes = num_classes
        self.D = D
        
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        
        # Stem: dense convolution
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers: sparse convolution
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6),
                spconv.SparseConv2d(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True)
            )
            self.downsample_layers.append(downsample_layer)
        
        # 4 feature resolution stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvBlock(dim=dims[i], drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (spconv.SubMConv2d, spconv.SparseConv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def upsample_mask(self, mask, scale):
        """
        Mask를 upsample
        
        Args:
            mask: [B, N] where N = p*p
            scale: upsampling factor
        """
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)

    def forward(self, x, mask):
        """
        Args:
            x: Input tensor [B, C, H, W]
            mask: Binary mask [B, N] where N = (H/patch_size) * (W/patch_size)
        
        Returns:
            Output tensor [B, C', H', W']
        """
        num_stages = len(self.stages)
        
        # Mask를 입력 해상도에 맞게 upsample
        mask = self.upsample_mask(mask, 2**(num_stages-1))        
        mask = mask.unsqueeze(1).type_as(x)  # [B, 1, H, W]
        
        # Patch embedding (dense)
        x = self.downsample_layers[0](x)
        x = x * (1. - mask)  # Masked positions를 0으로
        
        # Sparse encoding: dense → sparse
        x = to_sparse(x)
        
        # Sparse stages
        for i in range(4):
            if i > 0:
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        # Densify: sparse → dense
        x = x.dense()
        
        return x

convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}


# 사용 예시
if __name__ == "__main__":
    model = SparseConvNeXtV2(
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1,
        D=2
    )
    
    # 입력 생성
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Mask 생성 (예: 50% masking)
    num_patches = (224 // 4) ** 2  # stem이 4x downsampling
    mask = torch.rand(batch_size, num_patches) > 0.5
    mask = mask.float()
    
    # Forward
    output = model(x, mask)
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")