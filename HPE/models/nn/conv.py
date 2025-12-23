import torch
import torch.nn as nn
import spconv.pytorch as spconv

def to_sparse(x):
    """Dense tensor를 sparse tensor로 변환"""
    batch_size, in_channels, H, W = x.shape
    
    # Non-zero 위치 찾기
    mask = (x.abs().sum(dim=1) > 1e-6)  # [B, H, W]
    
    indices_list = []
    features_list = []
    
    for b in range(batch_size):
        nz_indices = torch.nonzero(mask[b], as_tuple=False)  # [N, 2] (y, x)
        
        if len(nz_indices) > 0:
            batch_indices = torch.full((len(nz_indices), 1), b, dtype=torch.int32, device=x.device)
            batch_nz_indices = torch.cat([batch_indices, nz_indices.int()], dim=1)
            indices_list.append(batch_nz_indices)
            
            y_coords = nz_indices[:, 0]
            x_coords = nz_indices[:, 1]
            feats = x[b, :, y_coords, x_coords].T  # [N, C]
            features_list.append(feats)
    
    if len(indices_list) > 0:
        indices = torch.cat(indices_list, dim=0)
        features = torch.cat(features_list, dim=0)
    else:
        indices = torch.zeros((0, 3), dtype=torch.int32, device=x.device)
        features = torch.zeros((0, in_channels), dtype=x.dtype, device=x.device)
    
    sparse_tensor = spconv.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=[H, W],
        batch_size=batch_size
    )
    
    return sparse_tensor

class SparseDepthwiseConv2d(nn.Module):
    """
    Sparse Depthwise Convolution
    spconv가 groups를 지원하지 않으므로 채널별로 개별 convolution 수행
    """
    def __init__(self, channels, kernel_size=7, padding=3, bias=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # 각 채널마다 개별 1x1 convolution (depthwise 효과)
        self.convs = nn.ModuleList([
            spconv.SubMConv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            for _ in range(channels)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """
        Args:
            x: spconv.SparseConvTensor with features [N, C]
        """
        if not isinstance(x, spconv.SparseConvTensor):
            raise TypeError("Input must be SparseConvTensor")
        
        # 각 채널별로 convolution 수행
        outputs = []
        for i in range(self.channels):
            # i번째 채널만 추출
            channel_features = x.features[:, i:i+1]
            
            # 해당 채널의 sparse tensor 생성
            channel_sparse = spconv.SparseConvTensor(
                features=channel_features,
                indices=x.indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            )
            
            # Convolution 적용
            out = self.convs[i](channel_sparse)
            outputs.append(out.features)
        
        # 모든 채널 결합
        new_features = torch.cat(outputs, dim=1)
        
        # Bias 추가
        if self.bias is not None:
            new_features = new_features + self.bias.view(1, -1)
        
        return x.replace_feature(new_features)
    

class EfficientSparseDepthwiseConv2d(nn.Module):
    """효율적인 Sparse Depthwise Convolution"""
    def __init__(self, channels, kernel_size=7, padding=3, bias=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # 7x7 depthwise를 위한 weight
        self.weight = nn.Parameter(torch.randn(channels, kernel_size * kernel_size))
        nn.init.kaiming_normal_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter('bias', None)
        
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
    
    def forward(self, x):
        if not isinstance(x, spconv.SparseConvTensor):
            raise TypeError("Input must be SparseConvTensor")
        
        # Dense로 변환
        dense = x.dense()  # [B, C, H, W]
        B, C, H, W = dense.shape
        
        # Unfold: [B, C*K*K, H*W]
        unfolded = self.unfold(dense)
        unfolded = unfolded.view(B, C, -1, H*W)  # [B, C, K*K, H*W]
        
        # Depthwise
        weight = self.weight.view(1, C, -1, 1)
        out = (unfolded * weight).sum(dim=2)  # [B, C, H*W]
        out = out.view(B, C, H, W)
        
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        
        # 다시 sparse로 변환
        return to_sparse(out)